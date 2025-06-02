#!/usr/bin/env python
import os
import time
import argparse
import torch
import deepspeed
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, hf_hub_download

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-Offload: Llama-3.2-1B Finetune on arXiv Abstracts")
    # Accept DeepSpeed's injected local_rank (even if unused directly)
    parser.add_argument("--local_rank", type=int, default=0,
        help="(DeepSpeed) Local rank passed automatically; no need to set manually."
    )
    # Core finetuning params (mirroring multigpu_torchrun.py)
    parser.add_argument("--num_epochs", type=int, required=True,
                        help="Number of epochs to run")
    parser.add_argument("--start_idx", type=int, required=True,
                        help="Starting sample index (inclusive)")
    parser.add_argument("--end_idx", type=int, required=True,
                        help="Ending sample index (exclusive)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Micro-batch size per GPU")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--initial_epoch", type=int, default=0,
                        help="Epoch to resume from (overrides checkpoint if any)")
    parser.add_argument("--hf_repo", type=str, required=True,
                        help="HF repo ID for saving checkpoints, e.g. username/repo-llama3-1B")
    parser.add_argument("--resume_file", type=str, default=None,
                        help="Name of a checkpoint file in HF repo to resume from, e.g. llama3.2-1B_0-1000-epoch-1.pt")
    # DeepSpeed config (we’ll dynamically create a dict, but allow override path if desired)
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="(Optional) Path to a DeepSpeed JSON config (if you want to override the built-in config)")
    return parser.parse_args()

class Trainer:
    def __init__(self, args, model_engine, tokenizer, dataloader, optimizer, device):
        self.args = args
        self.engine = model_engine
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device

        # Track how many samples we will process per epoch slice
        total_samples = args.end_idx - args.start_idx
        self.max_steps = (total_samples + 
                          (args.batch_size * args.accum_steps) - 1) // (args.batch_size * args.accum_steps)
        self.global_step = 0
        self.epochs_run = args.initial_epoch

        # If resuming from a checkpoint: deepseed will handle optimizer/scheduler states,
        # but we load the bare model weights into engine.module before deepspeed.initialize below.
        print(f"[Rank {self.engine.local_rank}] Ready to train from epoch {self.epochs_run}, max_steps={self.max_steps}")

    def save_checkpoint(self, epoch):
        """ Save a pure-model PyTorch .pt checkpoint and push to HF. """
        if self.engine.local_rank != 0:
            return
        disp_epoch = epoch + 1
        ckpt_name = f"llama3.2-1B_{self.args.start_idx}-{self.args.end_idx}-epoch-{disp_epoch}.pt"
        state_dict = self.engine.module.state_dict()
        torch.save({"MODEL_STATE": state_dict}, ckpt_name)
        print(f"[Rank {self.engine.local_rank}] Saved checkpoint {ckpt_name}")
        # Upload to HF
        if self.args.hf_repo:
            HfApi().upload_file(
                path_or_fileobj=ckpt_name,
                path_in_repo=ckpt_name,
                repo_id=self.args.hf_repo,
                repo_type="model"
            )
            print(f"[Rank {self.engine.local_rank}] Uploaded {ckpt_name} → HF repo {self.args.hf_repo}")

    def train(self):
        self.engine.train()
        for epoch in range(self.epochs_run, self.epochs_run + self.args.num_epochs):
            epoch_start = time.time()
            step_in_epoch = 0
            for batch in self.dataloader:
                # Move inputs to GPU
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward + backward + step via DeepSpeed engine
                outputs = self.engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                self.engine.backward(loss)
                self.engine.step()

                # Log to Weights & Biases (loss * accum_steps gives “actual” loss)
                if self.engine.local_rank == 0:
                    wandb.log({
                        "train_loss": loss.item() * self.args.accum_steps,
                        "epoch": epoch,
                        "step": self.engine.global_steps()
                    }, step=self.engine.global_steps())

                self.global_step += 1
                step_in_epoch += 1
                if self.global_step >= self.max_steps:
                    print(f"[Rank {self.engine.local_rank}] Reached max_steps {self.max_steps}, exiting early")
                    self.save_checkpoint(epoch)
                    return

            epoch_time = time.time() - epoch_start
            if self.engine.local_rank == 0:
                print(f"[Rank 0] Finished epoch {epoch} in {epoch_time:.1f}s; saving checkpoint…")
            self.save_checkpoint(epoch)

def main():
    args = parse_args()

    # -------------------------------
    # 1. Initialize Weights & Biases
    # -------------------------------
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
    else:
        # DeepSpeed will spawn one process per GPU; args.local_rank is injected by DeepSpeed
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    wandb.init(
        project="deepspeed-llama-3.2-1B-finetune",
        config={
            "num_epochs": args.num_epochs,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "initial_epoch": args.initial_epoch,
            "hf_repo": args.hf_repo,
            "resume_file": args.resume_file,
            "zero_stage": 2,
        }
    )
    if local_rank == 0:
        print("[Rank 0] W&B run initialized")

    # -------------------------------
    # 2. Load & Filter the Dataset
    # -------------------------------
    if local_rank == 0:
        print("[Rank 0] Loading and filtering dataset…")
    raw_ds = load_dataset("ash001/arxiv-abstract", split="train")
    # Filter out empty/whitespace-only lines
    raw_ds = raw_ds.filter(lambda example: example["text"].strip() != "")

    # Slice according to start_idx:end_idx
    total_len = len(raw_ds)
    if args.end_idx > total_len:
        raise ValueError(f"end_idx ({args.end_idx}) > dataset size ({total_len})")
    sliced = raw_ds.select(range(args.start_idx, args.end_idx))
    if local_rank == 0:
        print(f"[Rank 0] Selected slice [{args.start_idx}:{args.end_idx}) → {len(sliced)} samples")

    # -------------------------------
    # 3. Tokenize & Collate
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # Some Llama tokenizers require `padding_side="right"` and `use_fast=False`; adjust if needed.
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def tokenize_fn(ex):
        tok = tokenizer(
            ex["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = sliced.map(tokenize_fn, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )

    # -------------------------------
    # 4. Load Model & (Optional) Resume
    # -------------------------------
    if local_rank == 0:
        print("[Rank 0] Loading Llama-3.2-1B model…")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model.resize_token_embeddings(len(tokenizer))  # in case we added pad_token

    # If resuming from a checkpoint in HF repo:
    if args.resume_file:
        # Download via HF Hub
        if args.hf_repo:
            try:
                ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.resume_file)
            except Exception:
                ckpt_path = args.resume_file  # fallback to local path
        else:
            ckpt_path = args.resume_file

        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state.get("MODEL_STATE", state))
            if local_rank == 0:
                print(f"[Rank 0] Resumed model weights from {args.resume_file}")
        else:
            raise FileNotFoundError(f"Cannot find resume_file: {ckpt_path}")

    # Move model to device (DeepSpeed will re-wrap it)
    model.to(device)

    # -------------------------------
    # 5. Build the Optimizer
    # -------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # -------------------------------
    # 6. Construct DeepSpeed Config (Stage 2 + CPU Offload + FP16)
    # -------------------------------
    if args.deepspeed_config:
        # If user supplied a JSON path, DeepSpeed will load that instead:
        ds_config_path = args.deepspeed_config
        ds_config_dict = None
        if local_rank == 0:
            print(f"[Rank 0] Using external DeepSpeed config at {ds_config_path}")
    else:
        # Build an in-memory config dict
        ds_config_dict = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.accum_steps,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                # Note: Stage 3 would also offload parameters → add "offload_param" here
                "contiguous_gradients": True,
                "overlap_comm": True
            },
            "fp16": {
                "enabled": True
            }
        }
        ds_config_path = None
        if local_rank == 0:
            print("[Rank 0] Using built-in DeepSpeed ZeRO Stage 2 + CPU Offload + FP16 config")

    # -------------------------------
    # 7. Initialize DeepSpeed Engine
    # -------------------------------
    # We pass `config_params=ds_config_dict` if using the built-in config;
    # otherwise, we rely on `config=args.deepspeed_config` for DeepSpeed to load.
    if ds_config_dict is not None:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            config_params=ds_config_dict
        )
    else:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            config=ds_config_path
        )

    if local_rank == 0:
        print("[Rank 0] DeepSpeed Engine initialized")
        print(f"[Rank 0] World size: {model_engine.world_size}, "
              f"ZeRO stage: {model_engine.optimizer.zero_stage}")

    # -------------------------------
    # 8. Run Training
    # -------------------------------
    trainer = Trainer(args, model_engine, tokenizer, dataloader, optimizer, device)
    trainer.train()

    # -------------------------------
    # 9. Log Total Training Time to W&B
    # -------------------------------
    total_time = time.time() - wandb.run._start_time
    if local_rank == 0:
        wandb.log({"total_run_time_sec": total_time})
        print(f"[Rank 0] Total training time: {total_time:.1f}s")

    # DeepSpeed will handle dist.destroy_process_group() on exit automatically.

if __name__ == "__main__":
    main()
