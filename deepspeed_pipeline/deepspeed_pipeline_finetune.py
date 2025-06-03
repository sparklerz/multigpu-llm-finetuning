#!/usr/bin/env python
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

import wandb
from huggingface_hub import HfApi, hf_hub_download

# ───────────────────────────────────────────────────────────────────────────────
#  1. CONFIGURATION / HYPERPARAMETERS
# ───────────────────────────────────────────────────────────────────────────────

# Change to "meta-llama/Llama-3.2-1B-Instruct" if you want the 1B model.
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # <-- Use 1B if you don't mind single-GPU feasibility.
DATASET_NAME = "ash001/arxiv-abstract"
TARGET_SEQ_LEN = 512  # max token length per example
WARMUP_STEPS = 100
LEARNING_RATE = 1e-5

# ───────────────────────────────────────────────────────────────────────────────
#  2. ARGS PARSING
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="DeepSpeed Pipeline Parallelism: Fine-tune Llama with 2 GPUs + ZeRO2"
    )

    p.add_argument("--num_epochs", type=int, required=True,
                   help="Number of epochs to run")
    p.add_argument("--start_idx", type=int, required=True,
                   help="Dataset start index (inclusive)")
    p.add_argument("--end_idx", type=int, required=True,
                   help="Dataset end index (exclusive)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Micro-batch size per GPU (actual global batch = bs_per_gpu * world_size * accum_steps)")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="Gradient accumulation steps (micro-batches per step)")
    p.add_argument("--initial_epoch", type=int, default=0,
                   help="Epoch number to resume from (overrides checkpoint epoch in resume_file)")
    p.add_argument("--hf_repo", type=str, required=True,
                   help="Hugging Face repo ID to push checkpoints, e.g. 'username/my-llama-7b'")
    p.add_argument("--resume_file", type=str, default=None,
                   help="Checkpoint filename in the HF repo, e.g. 'checkpoint_epoch_1.pt'")

    # Any extra arguments after “--” in `deepspeed … script.py -- [args]` will land here.
    args = p.parse_args()
    return args

# ───────────────────────────────────────────────────────────────────────────────
#  3. SET UP DISTRIBUTED ENVIRONMENT (DeepSpeed)
# ───────────────────────────────────────────────────────────────────────────────

def ds_setup():
    # DeepSpeed will have already initialised torch.distributed via the launcher.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

# ───────────────────────────────────────────────────────────────────────────────
#  4. BUILD A PIPELINED MODEL
# ───────────────────────────────────────────────────────────────────────────────

class LlamaPipeModel(PipelineModule):
    """
    Wraps a Huggingface LlamaForCausalLM in a DeepSpeed PipelineModule.
    Splits the Transformer layers evenly into `num_stages` pieces.
    The final forward() returns the loss.
    """
    def __init__(self, model_name: str, num_stages: int):
        # 1) Load the HF model in CPU/GPU memory (we’ll let PipelineModule partition it later)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"  # This tells HF to place each submodule onto GPUs if available,
                               # but PipelineModule will override placement per stage.
        )

        # 2) Extract submodules in the correct order: embeddings → each block → final norm + lm_head
        #    For LlamaForCausalLM, the structure is roughly:
        #       hf_model.model.embed_tokens
        #       hf_model.model.layers (ModuleList of Transformer Decoder blocks)
        #       hf_model.model.norm
        #       hf_model.lm_head
        #
        #    We’ll break them out into a linear Python list of Layers.
        layers = []
        # Embedding
        layers.append(hf_model.model.embed_tokens)

        # Transformer Blocks
        for block in hf_model.model.layers:
            layers.append(block)

        # Final layer norm
        layers.append(hf_model.model.norm)

        # LM head (tie weights with embedding in HF, but we can include it as a standalone module)
        layers.append(hf_model.lm_head)

        # 3) Define a simple CrossEntropyLoss for causal LM. PipelineModule will know to append this
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        # 4) Call super() to let DeepSpeed partition these “layers” across `num_stages` GPUs.
        super().__init__(
            layers=layers,
            loss_fn=loss_fn,
            num_stages=num_stages,
            partition_method="parameters",  # “parameters” shards each layer’s weights; alternatives exist
            seed=42
        )

        # We also store the tokenizer’s config for later use (particularly position embeddings / config)
        self.config = hf_model.config

    def forward(self, input_ids, attention_mask, labels):
        """
        PipelineModule’s forward() automatically chops the input batch into micro-batches
        and passes them through each stage. We just need to define how to compute loss from logits.
        """
        # PipelineModule will run all “layers” in a sequence and the final output (call it logits)
        # will be a tensor of shape (batch_size, seq_len, vocab_size). We compute loss vs labels.
        logits = super().forward(input_ids, attention_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

# ───────────────────────────────────────────────────────────────────────────────
#  5. TRAINER CLASS (handles dataset slicing, dataloader, logging, checkpointing)
# ───────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self,
                 engine,
                 tokenizer,
                 dataloader,
                 start_idx: int,
                 end_idx: int,
                 hf_repo: str,
                 resume_file: str,
                 accum_steps: int,
                 initial_epoch: int):
        """
        engine       : the deepspeed engine returned by deepspeed.initialize()
        tokenizer    : the HF tokenizer
        dataloader   : DataLoader wrapping tokenized dataset
        start_idx    : slice start index (for checkpoint naming)
        end_idx      : slice end index
        hf_repo      : HuggingFace repo to push checkpoints
        resume_file  : name of the checkpoint file to resume (if any)
        accum_steps  : gradient accumulation steps
        initial_epoch: which epoch to start from (if resuming)
        """
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.engine = engine
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.hf_repo = hf_repo
        self.accum_steps = accum_steps

        # Compute how many optimizer steps we will do, so we can early-stop after max.
        total_samples = end_idx - start_idx
        world_size = dist.get_world_size()
        bs_per_gpu = dataloader.batch_size
        self.max_steps = (total_samples // (bs_per_gpu * world_size * accum_steps)) + 1

        # Epochs / global step tracking
        self.epoch = initial_epoch
        self.global_step = 0

        # If there is a resume_file (checkpoint), load it on rank 0, and broadcast to all
        if resume_file and self.local_rank == 0:
            try:
                ckpt_path = hf_hub_download(repo_id=hf_repo, filename=resume_file)
            except:
                ckpt_path = resume_file
            if os.path.exists(ckpt_path):
                print(f"[Rank {self.local_rank}] Loading checkpoint {resume_file}")
                engine.load_checkpoint(os.path.dirname(ckpt_path),
                                       tag=os.path.basename(ckpt_path).replace(".pt", ""))
        # barrier so everyone waits until model is loaded
        dist.barrier()

        # For W&B: rank 0 initialises W&B; other ranks skip
        if self.local_rank == 0:
            wandb.init(project="llama7b-pipeline-finetune",
                       name=f"slice_{start_idx}_{end_idx}",
                       config={
                           "model_name": MODEL_NAME,
                           "num_stages": 2,
                           "zero_stage": 2,
                           "batch_size_per_gpu": bs_per_gpu,
                           "accum_steps": accum_steps,
                           "learning_rate": LEARNING_RATE,
                           "max_epochs": None,  # we’ll log actual epochs
                           "start_idx": start_idx,
                           "end_idx": end_idx
                       })
            print(f"[Rank {self.local_rank}] WandB run initialized.")

    def save_checkpoint(self):
        """
        Save a checkpoint at the end of each epoch. Only rank 0 actually writes to disk
        and pushes to HF if needed.
        """
        if self.local_rank != 0:
            return
        tag = f"epoch_{self.epoch}"
        out_dir = f"ckpt_{self.start_idx}_{self.end_idx}_epoch_{self.epoch}"
        print(f"[Rank {self.local_rank}] Saving checkpoint → {out_dir} (tag={tag})")
        self.engine.save_checkpoint(out_dir, tag=tag)

        if self.hf_repo:
            # upload all files in the checkpoint folder to HF
            api = HfApi()
            for filename in os.listdir(out_dir):
                local_file = os.path.join(out_dir, filename)
                remote_path = os.path.join(out_dir, filename)
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=remote_path,
                    repo_id=self.hf_repo,
                    repo_type="model"
                )
            print(f"[Rank {self.local_rank}] Checkpoint {tag} uploaded to HF repo {self.hf_repo}")

    def train(self, num_epochs: int):
        """
        Main training loop over `num_epochs`. We slice the dataset via DistributedSampler’s set_epoch,
        do gradient accumulation, log to W&B, and checkpoint.
        """
        total_training_start = time.time()

        for ep in range(self.epoch, self.epoch + num_epochs):
            self.epoch = ep
            self.dataloader.sampler.set_epoch(ep)
            accum_counter = 0

            for step, batch in enumerate(self.dataloader):
                # Move batch to correct device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward + backward via DeepSpeed engine
                loss = self.engine(input_ids, attention_mask, labels)

                # DeepSpeed does loss_scaling internally in fp16; call backward via engine
                self.engine.backward(loss)
                accum_counter += 1

                # When accum_counter == accum_steps, we do one optimizer step
                if accum_counter == self.accum_steps:
                    self.engine.step()
                    self.global_step += 1
                    if self.local_rank == 0:
                        wandb.log({"train_loss": loss.item(), "step": self.global_step})
                    accum_counter = 0

                    # Early exit if we’ve done enough steps
                    if self.global_step >= self.max_steps:
                        break

            # Finish any leftover gradients
            if accum_counter > 0:
                self.engine.step()
                self.global_step += 1
                if self.local_rank == 0:
                    wandb.log({"train_loss": loss.item(), "step": self.global_step})

            # End of epoch: checkpoint + log wall-time
            epoch_time = time.time() - total_training_start
            if self.local_rank == 0:
                wandb.log({"epoch": self.epoch, "time_elapsed_sec": epoch_time})
                print(f"[Rank {self.local_rank}] Epoch {self.epoch} finished in {epoch_time:.2f}s")

            self.save_checkpoint()

            # Early-stop if max steps exceeded
            if self.global_step >= self.max_steps:
                break

        # After all epochs
        total_time = time.time() - total_training_start
        if self.local_rank == 0:
            wandb.log({"run_completed": True, "total_time_sec": total_time})
            print(f"[Rank {self.local_rank}] Training complete in {total_time:.2f}s")

        dist.barrier()  # ensure all ranks finish together


# ───────────────────────────────────────────────────────────────────────────────
#  6. MAIN
# ───────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    local_rank, world_size = ds_setup()

    # 1) Load & preprocess dataset slice
    if local_rank == 0:
        print(f"[Rank {local_rank}] Loading dataset {DATASET_NAME}…")
    ds = load_dataset(DATASET_NAME, split="train")

    # 2) Filter out empty lines
    ds = ds.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

    # 3) Slice [start_idx : end_idx]
    ds = ds.select(range(args.start_idx, args.end_idx))

    if local_rank == 0:
        print(f"[Rank {local_rank}] Filtered & sliced dataset has {len(ds)} samples.")

    # 4) Tokeniser & tokenisation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.model_max_length = TARGET_SEQ_LEN

    def tokenize_fn(ex):
        # We do causal LM: input_ids = tokenise(text); labels = same as input_ids
        toks = tokenizer(
            ex["text"],
            truncation=True,
            max_length=TARGET_SEQ_LEN,
            padding="max_length"
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tok_ds = ds.map(tokenize_fn, remove_columns=ds.column_names, batched=False)
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 5) DataLoader + DistributedSampler
    sampler = DistributedSampler(tok_ds, shuffle=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        tok_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        pin_memory=True
    )

    # 6) Build pipelined model
    #    We pass num_stages=2 so that DeepSpeed splits our list of layers evenly across 2 GPUs.
    pipeline_model = LlamaPipeModel(model_name=MODEL_NAME, num_stages=2)

    # 7) DeepSpeed config dict
    ds_config = {
        "train_batch_size": args.batch_size * world_size * args.accum_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "steps_per_print": 100,
        "wall_clock_breakdown": True,

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True
        },

        # No need to explicitly say “pipeline”: PipelineModule API handles that as long as we 
        # invoked deepspeed.initialize(model=PipelineModule, …) with --num_stages=2.
        #
        # If you want to force a particular partition method, you can also add:
        # "pipeline": {
        #     "partition_method": "parameters",
        #     "activation_checkpoint_interval": 0
        # }
        #
        # But the above is optional since PipelineModule already knows our partition_method.

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": LEARNING_RATE,
                "warmup_num_steps": WARMUP_STEPS
            }
        },

        "fp16": {
            "enabled": True,
            "loss_scale": 0,           # “auto” if set to 0
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }

    # 8) Initialize DeepSpeed engine
    engine, _, _, _ = deepspeed.initialize(
        model=pipeline_model,
        config_params=ds_config,
        model_parameters=[p for p in pipeline_model.parameters() if p.requires_grad]
    )

    if local_rank == 0:
        print(f"[Rank {local_rank}] DeepSpeed engine initialised with world_size={world_size}.")

    # 9) Load checkpoint if requested (DeepSpeed’s built-in load is called in Trainer)
    trainer = Trainer(
        engine=engine,
        tokenizer=tokenizer,
        dataloader=loader,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        hf_repo=args.hf_repo,
        resume_file=args.resume_file,
        accum_steps=args.accum_steps,
        initial_epoch=args.initial_epoch
    )

    # 10) Run training
    trainer.train(args.num_epochs)

    # 11) Clean up
    dist.barrier()
    if local_rank == 0:
        print(f"[Rank {local_rank}] Destroying process group; training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
