#!/usr/bin/env python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.nn import CrossEntropyLoss

import deepspeed
from deepspeed.pipe import PipelineModule
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

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

MODEL_NAME = "bigscience/bloomz-1b7"
DATASET_NAME = "ash001/arxiv-abstract"
TARGET_SEQ_LEN = 64  # max token length per example
WARMUP_STEPS = 100
LEARNING_RATE = 1e-5

# ───────────────────────────────────────────────────────────────────────────────
#  2. ARGS PARSING
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="DeepSpeed Pipeline Parallelism: Fine-tune Bloom with 2 GPUs + ZeRO1"
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
                   help="Hugging Face repo ID to push checkpoints, e.g. 'username/my-bloom-1.7b'")
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

class PassLabels(nn.Module):
    """Let a normal layer travel through the pipeline carrying labels with it."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args):
        packed = args
        while len(packed) == 1 and isinstance(packed[0], (tuple, list)):
            packed = packed[0]
        if len(packed) == 2:
            hidden, labels = packed
            out = self.layer(hidden)
        elif len(packed) == 4:
            # every transformer block gives (hidden, labels, alibi, pad_mask)
            hidden, labels, alibi, attention_mask = packed
            # final layer-norm only takes hidden
            if isinstance(self.layer, nn.LayerNorm):
                out = self.layer(hidden)
            else:
                out = self.layer(hidden, attention_mask=attention_mask, alibi=alibi)
        else:
            raise ValueError(f"Unexpected number of elements in packed: {len(packed)}")
        if isinstance(out, tuple):  # drop any extras (e.g. past_key_values)
            out = out[0]
        # carry everything forward
        if len(packed) == 2:
            return (out, labels)
        else:
            return (out, labels, alibi, attention_mask)

class EmbeddingBlock(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed = embed_tokens               # HF embedding layer

    def forward(self, *args):
        packed = args
        while len(packed) == 1 and isinstance(packed[0], (tuple, list)):
            packed = packed[0]
        if len(packed) != 4:
            raise ValueError(f"EmbeddingBlock expected 4 inputs, but got {len(packed)}")
        input_ids, labels, alibi, attention_mask = packed
        hidden = self.embed(input_ids)
        return (hidden, labels, alibi, attention_mask)

class LMHeadLossBlock(nn.Module):
    def __init__(self, lm_head, ignore_index=-100):
        super().__init__()
        self.lm_head = lm_head
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, *args):
        packed = args
        while len(packed) == 1 and isinstance(packed[0], (tuple, list)):
            packed = packed[0]
        if len(packed) == 4:
            hidden, labels, *_ = packed
        elif len(packed) == 2:
            hidden, labels = packed
        else:
            raise ValueError(f"Unexpected number of elements in packed: {len(packed)}")

        logits = self.lm_head(hidden)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        return loss

# ───────────────────────────────────────────────────────────────────────────────
#  4. BUILD A PIPELINED MODEL
# ───────────────────────────────────────────────────────────────────────────────

class BloomPipeModel(PipelineModule):
    """
    Wraps a Huggingface BloomForCausalLM in a DeepSpeed PipelineModule.
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
        #    For BloomForCausalLM, the structure is roughly:
        #       hf_model.model.embed_tokens
        #       hf_model.model.layers (ModuleList of Transformer Decoder blocks)
        #       hf_model.model.norm
        #       hf_model.lm_head
        #
        #    We’ll break them out into a linear Python list of Layers.
        layers = []
        # Embedding
        layers.append(EmbeddingBlock(hf_model.transformer.word_embeddings))

        # Transformer Blocks
        for block in hf_model.transformer.h:
            layers.append(PassLabels(block))

        layers.append(PassLabels(hf_model.transformer.ln_f))

        # LM head (tie weights with embedding in HF, but we can include it as a standalone module)
        layers.append(LMHeadLossBlock(hf_model.lm_head))

        # 4) Call super() to let DeepSpeed partition these “layers” across `num_stages` GPUs.
        super().__init__(
            layers=layers,
            num_stages=num_stages,
            partition_method="parameters"  # “parameters” shards each layer’s weights; alternatives exist
        )

        # We also store the tokenizer’s config for later use (particularly position embeddings / config)
        self.config = hf_model.config

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
            wandb.init(project="bloom-1.7b-pipeline-finetune",
                       name=f"slice_{start_idx}_{end_idx}",
                       config={
                           "model_name": MODEL_NAME,
                           "num_stages": dist.get_world_size(),
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
            data_iter = iter(self.dataloader)

            while self.global_step < self.max_steps:
                loss = self.engine.train_batch(data_iter=data_iter)
                if loss is None:               # iterator exhausted
                    break

                if self.engine.is_gradient_accumulation_boundary():
                    if self.local_rank == 0:
                        wandb.log({"train_loss": loss.item(),
                                   "step": self.global_step})
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
    num_stages = world_size

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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = TARGET_SEQ_LEN

    def batch_to_tuple(samples):
        batch = default_collate(samples)
        ids = batch["input_ids"].to(f"cuda:{local_rank}")
        lbls = batch["labels"].to(f"cuda:{local_rank}")
        attn = batch["attention_mask"].to(f"cuda:{local_rank}")
        n_head = pipeline_model.config.n_head
        alibi = build_alibi_tensor(attn, n_head, dtype=torch.float16).to(f"cuda:{local_rank}")
        pad_mask = ~attn.to(torch.bool)
        return (ids, lbls, alibi, pad_mask)

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
        pin_memory=True,
        collate_fn=batch_to_tuple
    )

    # 6) Build pipelined model
    #    We pass num_stages=2 so that DeepSpeed splits our list of layers evenly across 2 GPUs.
    pipeline_model = BloomPipeModel(model_name=MODEL_NAME, num_stages=num_stages)

    # 7) DeepSpeed config dict
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "steps_per_print": 100,
        "wall_clock_breakdown": True,

        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e6,
            "reduce_scatter": False,
            "reduce_bucket_size": 5e6,
            "overlap_comm": False,
            "contiguous_gradients": False,
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
        },
        "activation_checkpointing": {
            "partition_activations": True,          # each stage keeps only its slice
            "contiguous_memory_optimization": True, # fewer cudaMallocs
            "cpu_checkpointing": False,              # stay on-GPU, faster
            "checkpoint_in_training": True
        }
    }

    engine, _, _, _ = deepspeed.initialize(
        model=pipeline_model,
        config_params=ds_config,
        model_parameters=[p for p in pipeline_model.parameters() if p.requires_grad]
    )

    if local_rank == 0:
        print(f"[Rank {local_rank}] DeepSpeed engine initialised with world_size={world_size}.")

    # 9) Load checkpoint if requested
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
