#!/usr/bin/env python
import os
import time
import torch
import torch.nn as nn
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
    AutoModelForCausalLM
)
from datasets import load_dataset

import wandb
from huggingface_hub import HfApi, hf_hub_download

# ───────────────────────────────────────────────────────────────────────────────
#  1. CONFIGURATION / HYPERPARAMETERS
# ───────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-2-2b-it"
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
        description="DeepSpeed Pipeline Parallelism: Fine-tune Gemma with 2 GPUs + ZeRO1"
    )
    p.add_argument("--local_rank", type=int, default=0,
                   help="(DeepSpeed) Local rank, passed automatically")
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
                   help="Hugging Face repo ID to push checkpoints, e.g. 'username/my-gemma-2b'")
    p.add_argument("--resume_file", type=str, default=None,
                   help="Checkpoint filename in the HF repo, e.g. 'checkpoint_epoch_1.pt'")

    # Any extra arguments after “--” in `deepspeed … script.py -- [args]` will land here.
    args = p.parse_args()
    return args

# ───────────────────────────────────────────────────────────────────────────────
#  3. SET UP DISTRIBUTED ENVIRONMENT (DeepSpeed)
# ───────────────────────────────────────────────────────────────────────────────

def ds_setup():
    deepspeed.init_distributed()  # calls torch.distributed.init_process_group under the hood
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

# ───────────────────────────────────────────────────────────────────────────────
#  4. BUILD A PIPELINED MODEL
# ───────────────────────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int,
                     head_dim: int,
                     device: torch.device,
                     dtype: torch.dtype,
                     theta: float = 10000.0):
    """
    Return (cos, sin) tensors with shape [seq_len, head_dim].

    Equivalent to the helper that used to live in
    `transformers.models.gemma2.modeling_gemma2`.
    """
    # half of head_dim because we interleave even/odd
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                             device=device,
                                             dtype=dtype) / head_dim))
    # [seq_len, head_dim // 2]
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)               # [seq_len, head_dim]
    return emb.cos(), emb.sin()

class GemmaPipeModel(PipelineModule):
    """
    Wraps a Huggingface GemmaForCausalLM in a DeepSpeed PipelineModule.
    Splits the Transformer layers evenly into `num_stages` pieces.
    The final forward() returns the loss.
    """
    def __init__(self, model_name: str, num_stages: int):
        # 1) Load the HF model in CPU/GPU memory (we’ll let PipelineModule partition it later)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None
        )

        # 2) Extract submodules in the correct order: embeddings → each block → final norm + lm_head
        #    We’ll break them out into a linear Python list of Layers.
        layers = []
        # Embedding
        layers.append(GemmaInputStage(hf_model))

        # Transformer Blocks
        for block in hf_model.model.layers:
            layers.append(GemmaDecoderWrapper(block, theta=getattr(hf_model.config, "rope_theta", 10000.0)))

        # Final layer norm
        layers.append(GemmaNormWrapper(hf_model.model.norm))

        # LM head (tie weights with embedding in HF, but we can include it as a standalone module)
        layers.append(hf_model.lm_head)

        # 3) Define a simple CrossEntropyLoss for causal LM. PipelineModule will know to append this
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        # 4) Call super() to let DeepSpeed partition these “layers” across `num_stages` GPUs.
        super().__init__(
            layers=layers,
            loss_fn=loss_fn,
            num_stages=num_stages,
            partition_method="parameters"  # “parameters” shards each layer’s weights; alternatives exist
        )

        # We also store the tokenizer’s config for later use (particularly position embeddings / config)
        self.config = hf_model.config

class GemmaInputStage(nn.Module):
    """token embeddings → hidden_states"""
    def __init__(self, hf):
        super().__init__()
        self.embed_tokens = hf.model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)          # [B, L, E]


class GemmaDecoderWrapper(nn.Module):
    """Adds RoPE cache that exactly matches this layer’s head_dim."""
    def __init__(self, layer, theta=10000.0):
        super().__init__()
        self.layer = layer
        self.theta = theta
        # grab static attributes once
        self.head_dim = layer.self_attn.head_dim       # right size (256 here)

    def forward(self, hidden):
        seq_len = hidden.size(1)
        cos, sin = build_rope_cache(
            seq_len,
            self.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
            theta=self.theta,
        )
        return self.layer(hidden, position_embeddings=(cos, sin))
        # still returns hidden_states


class GemmaNormWrapper(nn.Module):
    """final layer-norm before lm_head"""
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, hidden):
        return self.norm(hidden)

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
            wandb.init(project="gemma2b-pipeline-finetune",
                       name=f"slice_{start_idx}_{end_idx}",
                       config={
                           "model_name": MODEL_NAME,
                           "num_stages": 2,
                           "zero_stage": 1,
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

            for step, batch in enumerate(self.dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                micro = (batch["input_ids"], batch["labels"])

                loss = self.engine.train_batch(iter([micro]))

                if self.local_rank == 0:
                    wandb.log({"train_loss": loss.item(), "step": self.global_step})
                
                self.global_step += 1
                
                if self.global_step >= self.max_steps:
                    break

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    loader = DataLoader(
        tok_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True
    )

    # 6) Build pipelined model
    #    We pass num_stages=2 so that DeepSpeed splits our list of layers evenly across 2 GPUs.
    pipeline_model = GemmaPipeModel(model_name=MODEL_NAME, num_stages=2)

    # 7) DeepSpeed config dict
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "steps_per_print": 100,
        "wall_clock_breakdown": True,

        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "none"
            }
        },

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

        "pipeline": {
            "pipe_partitioned": True,
            "grad_partitioned": True
        },

        "fp16": {
            "enabled": True,
            "loss_scale": 0
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
