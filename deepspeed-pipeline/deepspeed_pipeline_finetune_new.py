import os, math, time, argparse, torch, torch.nn as nn, torch.distributed as dist
import torch.nn.functional as F
import deepspeed
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.pipe import PipelineModule, LayerSpec

# ---------- helpers ---------------------------------------------------------
def get_position_ids(seq_len, device):
    return torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

class EmbeddingPipe(nn.Module):
    """Embeddings + position encodings for OPT."""
    def __init__(self, decoder):
        super().__init__()
        self.embed_tokens      = decoder.embed_tokens
        self.embed_positions   = decoder.embed_positions
        self.project_in        = decoder.project_in

    def forward(self, inputs):
        ids, attn, labels = inputs
        pos_ids = get_position_ids(ids.size(1), ids.device)
        hidden  = self.embed_tokens(ids) + self.embed_positions(pos_ids)
        if self.project_in is not None:
            hidden = self.project_in(hidden)
        return hidden, attn, labels

class DecoderLayerPipe(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, inputs):
        hidden, attn, labels = inputs
        hidden = self.layer(hidden, attention_mask=attn)[0]
        return hidden, attn, labels

class FinalNormPipe(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, inputs):
        hidden, attn, labels = inputs
        return self.norm(hidden), attn, labels

class LMHeadPipe(nn.Module):
    """Last stage: computes logits and loss."""
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
    def forward(self, inputs):
        hidden, _, labels = inputs
        logits   = self.lm_head(hidden)
        # shift-left language-model loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index = -100
            )
        return loss

def build_pipeline(model):
    """Turn HF OPT into a 2-stage PipelineModule."""
    dec = model.model.decoder
    layers = []

    # stage-1: embeddings + 12 decoder layers
    layers.append(LayerSpec(EmbeddingPipe, dec))
    for l in dec.layers[:12]:
        layers.append(LayerSpec(DecoderLayerPipe, l))

    # stage-2: remaining decoder layers + final norm + lm_head + loss
    for l in dec.layers[12:]:
        layers.append(LayerSpec(DecoderLayerPipe, l))
    layers.append(LayerSpec(FinalNormPipe, dec.final_layer_norm))
    layers.append(LayerSpec(LMHeadPipe, model.lm_head))

    return PipelineModule(
        layers          = layers,
        loss_fn         = None,          # handled in LMHeadPipe
        num_stages      = 2,
        partition_method= "uniform",     # split 50/50
        activation_checkpoint_interval = 0
    )

def filter_empty(example):            # drop blank abstracts early
    return example["text"].strip() != ""

# ---------- training loop ---------------------------------------------------
def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    rank      = int(os.environ["RANK"])
    local_rank= int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # --- WANDB (single process logs) ----------------------------------------
    if rank == 0:
        wandb.init(project="opt1.3b-pipeline", config=vars(args))

    # --- dataset ------------------------------------------------------------
    raw_ds  = load_dataset("ash001/arxiv-abstract", split="train")
    raw_ds  = raw_ds.filter(filter_empty)
    raw_ds  = raw_ds.select(range(args.start_idx, args.end_idx))

    tok     = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    tok.pad_token = tok.eos_token

    def tokenize(ex):
        out = tok(ex["text"],
                  truncation=True,
                  max_length=512,
                  padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = raw_ds.map(tokenize, remove_columns=raw_ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size  = args.batch_size,   # micro-batch per GPU
        shuffle     = True,
        pin_memory  = True,
    )

    # --- model & DeepSpeed --------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        torch_dtype = torch.float16
    )
    pipe_model = build_pipeline(base_model)

    ds_config = {
        "fp16": {"enabled": True},
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "pipeline":      {"seed_layers": False},
        "pipeline_parallel_size": 2,
        "zero_optimization": {
            "stage": 1,                        # ZeRO-1
            "offload_optimizer": {"device": "none"}
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": True
    }

    engine, optimizer, _, _ = deepspeed.initialize(
        model               = pipe_model,
        model_parameters    = [p for p in pipe_model.parameters() if p.requires_grad],
        config              = ds_config
    )

    # --- training -----------------------------------------------------------
    global_steps   = 0
    samples_target = args.end_idx - args.start_idx
    samples_seen   = 0
    t0             = time.time()

    for epoch in range(args.initial_epoch, args.initial_epoch + args.num_epochs):
        for batch in loader:
            # pipeline expects tuple (ids, mask, labels)
            input_ids      = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels         = batch["labels"].cuda()

            loss = engine((input_ids, attention_mask, labels))
            engine.backward(loss)
            engine.step()

            samples_seen += input_ids.size(0)
            global_steps += 1

            if rank == 0 and global_steps % 10 == 0:
                wandb.log({"train_loss": loss.item(),
                           "samples_seen": samples_seen,
                           "step": global_steps})

            if samples_seen >= samples_target:
                break
        if samples_seen >= samples_target:
            break

    # --- finish -------------------------------------------------------------
    if rank == 0:
        elapsed = time.time() - t0
        wandb.log({"total_training_time_sec": elapsed})
        print(f"Finished slice {args.start_idx}-{args.end_idx} in {elapsed/60:.2f} min")
        # push tokenizer + final engine weights if desired
        if args.hf_repo:
            tok.push_to_hub(args.hf_repo)
            engine.save_checkpoint(".", tag="pipeline_last")
            from huggingface_hub import HfApi
            HfApi().upload_folder(folder_path="pipeline_last",
                                  repo_id=args.hf_repo,
                                  repo_type="model")

# ---------- CLI -------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dual-GPU DeepSpeed pipeline fine-tune OPT-1.3B")
    p.add_argument("--local_rank",    type=int, default=-1)
    p.add_argument("--num_epochs",    type=int, required=True)
    p.add_argument("--start_idx",     type=int, required=True)
    p.add_argument("--end_idx",       type=int, required=True)
    p.add_argument("--batch_size",    type=int, default=1)
    p.add_argument("--accum_steps",   type=int, default=1)
    p.add_argument("--initial_epoch", type=int, default=0)
    p.add_argument("--hf_repo",       type=str, required=True)
    p.add_argument("--resume_file",   type=str)
    args = p.parse_args()
    main(args)
