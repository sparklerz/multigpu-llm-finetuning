import os, math, time, argparse, torch, torch.nn as nn, torch.distributed as dist
import torch.nn.functional as F
import deepspeed
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from itertools import repeat
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# ---------- helpers ---------------------------------------------------------

def normalise_batch(inputs):
    # 1) peel off DeepSpeed’s wrapper(s)
    while isinstance(inputs, (tuple, list)) and len(inputs) == 1:
        inputs = inputs[0]

    # 2) convert to canonical (ids, attn, labels) --------------------------
    if isinstance(inputs, dict):                # dict from a DataLoader
        ids   = inputs["input_ids"]
        attn  = inputs.get("attention_mask", None)
        labels= inputs.get("labels", None)

    elif torch.is_tensor(inputs):               # lone ids tensor
        ids   = inputs
        attn  = None
        labels= None

    elif isinstance(inputs, (tuple, list)):
        if len(inputs) == 3:                    # the good case
            ids, attn, labels = inputs
        elif len(inputs) == 2:                  # (ids, attn) but no labels
            ids, attn = inputs
            labels    = None
        else:                                   # e.g. (ids,)
            ids      = inputs[0]
            attn     = None
            labels   = None
    else:
        raise TypeError(f"Unexpected micro-batch type: {type(inputs)}")

    # 3) final sanity-check
    return ids, attn, labels

def build_position_ids(batch):
    bsz, seq = batch.shape[:2]
    return torch.arange(seq, device=batch.device).unsqueeze(0).expand(bsz, -1)


class EmbeddingPipe(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.embed_tokens    = decoder.embed_tokens
        self.hidden_size = decoder.embed_tokens.embedding_dim

    def forward(self, inputs):
        ids, attn, labels = normalise_batch(inputs)
        if attn is None:
            attn = (ids != self.embed_tokens.padding_idx).long()
        hidden = self.embed_tokens(ids)
        return hidden, attn, labels

class DecoderLayerPipe(nn.Module):
    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, inputs):
        hidden, attn, labels = normalise_batch(inputs)
        if attn is not None and attn.dim() == 2:          # (B, S)
            attn = _prepare_4d_causal_attention_mask(
                        attn.to(torch.bool),
                        hidden.shape[:2],                       # (batch, seq_len)
                        hidden,                                 # embeds (dtype/device)
                        past_key_values_length=0,               # no KV-cache in training
                    )
        pos_ids          = build_position_ids(hidden)
        pos_embeddings   = self.rotary_emb(hidden, pos_ids)
        hidden = self.layer(hidden, attention_mask=attn, position_ids = pos_ids, position_embeddings = pos_embeddings,)[0]
        return hidden, attn, labels

class FinalNormPipe(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, inputs):
        hidden, attn, labels = normalise_batch(inputs)
        return self.norm(hidden), attn, labels

class LMHeadPipe(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
    def forward(self, inputs):
        hidden, _attn, labels = normalise_batch(inputs)
        logits = self.lm_head(hidden)
        vocab = logits.size(-1)
        if labels is not None:
            max_lbl = int(labels.max().item())
            if not -100 <= max_lbl < logits.size(-1):
                raise RuntimeError(
                    f"labels contain id {max_lbl} ≥ vocab_size={logits.size(-1)}"
                )
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
            return loss
        return None

def build_pipeline(model):
    """Turn HF Llama into a 2-stage PipelineModule."""
    dec = model.model
    try:
        rope = dec.layers[0].self_attn.rotary_emb
    except AttributeError:
        rope = LlamaRotaryEmbedding(model.config)
        
    n_layers = len(dec.layers)
    split_point  = n_layers // 2
    layers = []
    norm_layer = getattr(dec, "norm", None)

    # stage-1: embeddings + n_layers // 2 decoder layers
    layers.append(LayerSpec(EmbeddingPipe, dec))
    for l in dec.layers[:split_point]:
        layers.append(LayerSpec(DecoderLayerPipe, l, rope))

    # stage-2: remaining decoder layers + final norm + lm_head + loss
    for l in dec.layers[split_point:]:
        layers.append(LayerSpec(DecoderLayerPipe, l, rope))
    layers.append(LayerSpec(FinalNormPipe, norm_layer))
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
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    rank = dist.get_rank()

    # --- WANDB (single process logs) ----------------------------------------
    if rank == 0:
        wandb.init(project="llama-1b-ds-pipeline", config=vars(args))

    # --- dataset ------------------------------------------------------------
    raw_ds  = load_dataset("ash001/arxiv-abstract", split="train")
    raw_ds  = raw_ds.filter(filter_empty)
    raw_ds  = raw_ds.select(range(args.start_idx, args.end_idx))

    tok     = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16
    )

    base_model.resize_token_embeddings(len(tok))

    pad_id = tok.pad_token_id
    base_model.config.pad_token_id = pad_id
    base_model.model.embed_tokens.padding_idx = pad_id

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
    dist.barrier()

    # --- training -----------------------------------------------------------
    global_steps   = 0
    samples_target = args.end_idx - args.start_idx
    samples_seen   = 0
    t0             = time.time()

    for epoch in range(args.initial_epoch, args.initial_epoch + args.num_epochs):

        if engine.is_first_stage():
            data_stream = (
                (batch["input_ids"].cuda(non_blocking=True),
                batch["attention_mask"].cuda(non_blocking=True),
                batch["labels"].cuda(non_blocking=True))
                for batch in loader
            )
        else:
            data_stream = repeat(None)            # dummy iterator – never consumed

        while True:
            # first stage pushes micro-batches; other stages just drive the pipe
            try:
                loss = (engine.train_batch(data_iter=data_stream)
                        if engine.is_first_stage()
                        else engine.train_batch())
            except StopIteration:
                break                             # data_stream exhausted — epoch done


            samples_seen += args.batch_size * args.accum_steps

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
    p = argparse.ArgumentParser(description="Dual-GPU DeepSpeed pipeline fine-tune Llama-1B")
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
