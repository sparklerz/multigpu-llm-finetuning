import os
import time
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import wandb

# FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "ash001/arxiv-abstract"


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size


def auto_wrap_policy(module):
    # Wrap transformer decoder layers
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    return isinstance(module, LlamaDecoderLayer)


class Trainer:
    def __init__(
        self,
        num_epochs,
        start_idx,
        end_idx,
        batch_size,
        accum_steps,
        initial_epoch,
        hf_repo,
        resume_file
    ):
        self.local_rank, self.world_size = ddp_setup()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.num_epochs = num_epochs
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.accum_steps = accum_steps
        self.hf_repo = hf_repo
        self.initial_epoch = initial_epoch
        self.global_step = 0
        self.epochs_run = initial_epoch

        # Load and prepare dataset slice
        ds = load_dataset(DATASET_NAME, split="train")
        ds = ds.select(range(start_idx, end_idx))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        def tokenize_fn(example):
            tok = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
            tok['labels'] = tok['input_ids'].copy()
            return tok
        tok_ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
        tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        sampler = DistributedSampler(tok_ds, shuffle=True)
        self.loader = DataLoader(tok_ds, batch_size=batch_size, sampler=sampler,
                                 collate_fn=collator, pin_memory=True)

        # Initialize model with FSDP
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        model.gradient_checkpointing_enable()
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            ),
            device_id=self.local_rank
        )
        self.model = fsdp_model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Compute total steps
        total_samples = end_idx - start_idx
        self.max_steps = math.ceil(total_samples / (batch_size * self.world_size * accum_steps))

        # Resume checkpoint if specified
        if hf_repo and resume_file and self.local_rank == 0:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=resume_file)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['MODEL_STATE'])
            self.global_step = ckpt.get('GLOBAL_STEP', 0)
            print(f"[Rank {self.local_rank}] Resumed from {resume_file} at step {self.global_step}")
        dist.barrier()

        # Initialize W&B on rank 0
        if self.local_rank == 0:
            wandb.init(
                project="llama-fsdp-arxiv",
                config={
                    "num_epochs": num_epochs,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "batch_size": batch_size,
                    "accum_steps": accum_steps,
                    "initial_epoch": initial_epoch,
                    "hf_repo": hf_repo
                }
            )
            wandb.run.name = f"fsdp-llama3.2-{wandb.run.id}"
            # Watch model parameters and gradients
            wandb.watch(self.model.module if hasattr(self.model, 'module') else self.model,
                        log="all", log_freq=10)

    def _save_checkpoint(self):
        if self.local_rank != 0:
            return
        epoch = self.epochs_run + 1
        name = f"llama3.2_1B_{self.start_idx}-{self.end_idx}-epoch-{epoch}.pt"
        state = {
            "MODEL_STATE": self.model.state_dict(),
            "GLOBAL_STEP": self.global_step,
            "EPOCHS_RUN": self.epochs_run
        }
        torch.save(state, name)
        print(f"[Rank {self.local_rank}] Saved checkpoint {name}")
        if self.hf_repo:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=name,
                                path_in_repo=name,
                                repo_id=self.hf_repo,
                                repo_type="model")
        # Log as W&B artifact
        artifact = wandb.Artifact(
            name=f"model-epoch-{epoch}",
            type="model",
            description=f"Checkpoint at epoch {epoch}"
        )
        artifact.add_file(name)
        wandb.log_artifact(artifact)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs_run, self.epochs_run + self.num_epochs):
            self.loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            accum_counter = 0
            for batch in self.loader:
                labels = batch['labels'].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                loss = self.model(**inputs, labels=labels).loss

                # Batch-level logging
                if self.local_rank == 0:
                    wandb.log({
                        "train_loss": loss.item() * self.accum_steps,
                        "step": self.global_step
                    })

                # Accumulate for epoch average
                epoch_loss += loss.item() * self.accum_steps
                loss = loss / self.accum_steps
                loss.backward()
                accum_counter += 1

                if accum_counter == self.accum_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    accum_counter = 0
                    if self.local_rank == 0:
                        print(f"Completed step {self.global_step}/{self.max_steps}")
                if self.global_step >= self.max_steps:
                    break

            # Final optimizer step if needed
            if accum_counter > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Epoch-level logging
            avg_loss = epoch_loss / (self.max_steps if self.max_steps else 1)
            if self.local_rank == 0:
                wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

            self.epochs_run += 1
            self._save_checkpoint()
            dist.barrier()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--initial_epoch", type=int, default=0)
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--resume_file", type=str, default=None)
    args = parser.parse_args()

    trainer = Trainer(
        num_epochs=args.num_epochs,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        initial_epoch=args.initial_epoch,
        hf_repo=args.hf_repo,
        resume_file=args.resume_file
    )
    trainer.train()

    # Finish W&B run
    if trainer.local_rank == 0:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
