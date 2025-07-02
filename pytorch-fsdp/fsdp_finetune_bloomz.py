import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import wandb
import functools

# FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, FullStateDictConfig, StateDictType, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.bloom.modeling_bloom import BloomBlock

# Make sure each process only uses one OMP thread (avoids NCCL warnings).
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_NAME = "bigscience/bloomz-1b7"
DATASET_NAME = "ash001/arxiv-abstract"


def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    return local_rank, world_size


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
        torch.backends.cudnn.benchmark = True

        self.num_epochs = num_epochs
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.initial_epoch = initial_epoch
        self.hf_repo = hf_repo
        self.resume_file = resume_file
        self.global_step = 0
        self.epochs_run = initial_epoch

        # Load and prepare dataset slice
        ds = load_dataset(DATASET_NAME, split="train")
        ds = ds.select(range(start_idx, end_idx))
        ds = ds.filter(lambda ex: ex["text"] and ex["text"].strip())

        # Tokenizer setup (ensure pad_token exists)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        def tokenize_fn(example):
            tok = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length", return_attention_mask=True)
            # Replace pad token labels with -100 to ignore in loss
            labels = tok["input_ids"].copy()
            labels = [lbl if lbl != tokenizer.pad_token_id else -100 for lbl in labels]
            tok["labels"] = labels
            return tok

        tok_ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
        tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        sampler = DistributedSampler(tok_ds, shuffle=True)
        self.loader = DataLoader(
            tok_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            pin_memory=True
        )

        # Initialize model with FSDP
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={BloomBlock},
        )
        
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ),
            device_id=self.local_rank
        )
        self.model = fsdp_model.to(self.device)

        # Optimizer and GradScaler for AMP
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)
        self.scaler = torch.amp.GradScaler()

        # Compute total steps
        total_samples = end_idx - start_idx
        self.max_steps = math.ceil(total_samples / (batch_size * self.world_size * accum_steps))

        # Resume checkpoint if specified
        if hf_repo and resume_file and self.local_rank == 0:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=resume_file)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["MODEL_STATE"])
            self.global_step = ckpt.get("GLOBAL_STEP", 0)
            self.epochs_run = ckpt.get("EPOCHS_RUN", initial_epoch)
            print(f"[Rank {self.local_rank}] Resumed from {resume_file} at step {self.global_step}")
        dist.barrier()

        # Initialize W&B on rank 0
        if self.local_rank == 0:
            wandb.init(
                project="bloomz-1.7B-fsdp-arxiv",
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
            wandb.run.name = f"fsdp-bloomz-{wandb.run.id}"
            # Watch model parameters and gradients
            wandb.watch(self.model.module if hasattr(self.model, 'module') else self.model,
                        log="all", log_freq=10)

    def _save_checkpoint(self):
        full_state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_cfg):
            sd = self.model.state_dict()

        if self.local_rank == 0:
            epoch = self.epochs_run
            name = f"bloomz_1.7B_{self.start_idx}-{self.end_idx}-epoch-{epoch}.pt"
            torch.save({
                "MODEL_STATE": sd,
                "GLOBAL_STEP": self.global_step,
                "EPOCHS_RUN": self.epochs_run
            }, name)
            print(f"[Rank {self.local_rank}] Saved checkpoint {name}")
            if self.hf_repo:
                from huggingface_hub import HfApi
                HfApi().upload_file(
                    path_or_fileobj=name,
                    path_in_repo=name,
                    repo_id=self.hf_repo,
                    repo_type="model"
                )
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
        for ep in range(self.epochs_run, self.epochs_run + self.num_epochs):
            self.loader.sampler.set_epoch(ep)
            epoch_step_loss = 0.0
            steps_this_epoch = 0
            accum = 0
            step_loss = 0.0

            for microbatch_idx, batch in enumerate(self.loader):
                # Send data to the local GPU
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    raw_loss = outputs.loss
                    scaled_loss = raw_loss / self.accum_steps

                # Check for NaN and print exactly which microbatch it occurred
                if torch.isnan(raw_loss):
                    print(
                        f"[Rank {self.local_rank}] *** raw_loss is NaN at "
                        f"epoch {ep}, microbatch {microbatch_idx}, "
                        f"accum-step {accum}, global_step {self.global_step} ***"
                    )

                step_loss += raw_loss.item()

                # Backward pass with scaling
                self.scaler.scale(scaled_loss).backward()
                accum += 1

                if accum == self.accum_steps:
                    # Unscale, clip gradients, optimizer step
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # Compute averaged loss for this optimizer step
                    avg_step_loss = step_loss / self.accum_steps
                    epoch_step_loss += avg_step_loss
                    steps_this_epoch += 1
                    self.global_step += 1
                    accum = 0
                    step_loss = 0.0

                    # Dynamic logging per step
                    print(f"[Rank {self.local_rank}] Step {self.global_step} | Avg Loss: {avg_step_loss:.4f}")
                    if self.local_rank == 0:
                        wandb.log({"train_loss": avg_step_loss}, step=self.global_step)

                if self.global_step >= self.max_steps:
                    break

            # Handle leftover accumulation
            if accum > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Compute averaged leftover loss
                avg_leftover_loss = step_loss / accum
                epoch_step_loss += avg_leftover_loss
                steps_this_epoch += 1
                self.global_step += 1
                print(f"[Rank {self.local_rank}] Step {self.global_step} | Avg Loss: {avg_leftover_loss:.4f}")
                if self.local_rank == 0:
                    wandb.log({"train_loss": avg_leftover_loss}, step=self.global_step)

            epoch_avg_loss = epoch_step_loss / max(steps_this_epoch, 1)

            loss_tensor = torch.tensor(epoch_avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_avg_loss = loss_tensor.item() / self.world_size

            print(f"[Rank {self.local_rank}] Epoch {ep+1} completed | Avg Loss: {epoch_avg_loss:.4f}")
            if self.local_rank == 0:
                wandb.log({"epoch": ep + 1, "avg_loss": epoch_avg_loss})

            self.epochs_run += 1
            self._save_checkpoint()
            dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
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