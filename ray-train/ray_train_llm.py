import os, time, torch, ray
from datasets import load_dataset
from ray.train.huggingface.transformers import TransformersTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          TrainingArguments,
                          TrainerCallback)
from huggingface_hub import Repository
import wandb

HF_REPO = "ash001/ray-train-zero-3-bloom-3B"

# ────────────────────────────────────────────────
# 0  Simple W&B time-tracking callback
class WallClockCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            self._start = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            wandb.log({"train/runtime_seconds": time.time() - self._start})

class HubTagEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kw):
        if torch.distributed.get_rank() == 0:
            repo = Repository(args.output_dir, clone_from=HF_REPO, token=os.environ.get("HF_TOKEN"))
            tag = f"epoch-{int(state.epoch)}"
            repo.git_tag(tag)
            repo.git_push(tags=True)


# ────────────────────────────────────────────────
# 1  Ray “per-worker” initialiser
def trainer_init_per_worker(train_dataset=None, eval_dataset=None, **cfg):
    model_name = cfg["model_name"]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tok   = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token        # safety for causal models
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir           = "./outputs",
        evaluation_strategy  = "steps",
        eval_steps           = 500,
        per_device_train_batch_size = cfg["per_device_batch"],
        per_device_eval_batch_size  = cfg["per_device_batch"],
        learning_rate        = cfg["lr"],
        num_train_epochs     = cfg["epochs"],
        logging_steps        = 50,
        gradient_accumulation_steps = cfg["grad_accum"],
        gradient_checkpointing       = True,
        save_strategy        = "epoch",
        push_to_hub          = True,
        hub_model_id         = HF_REPO,
        hub_strategy         = "checkpoint",
        hub_private_repo     = False,
        deepspeed           = cfg["ds_config_path"],
        fp16                = True,
        report_to           = "wandb",
        run_name            = cfg["wandb_run"],
    )

    return dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[WallClockCallback(), HubTagEpochCallback()],
        tokenizer=tok
    )


# ────────────────────────────────────────────────
# 2  Dataset: IMDB
def get_dataset(tok):
    ds = load_dataset("imdb", split="train[:2%]")
    ds = ds.train_test_split(test_size=0.1)
    def tok_fn(ex): return tok(ex["text"], truncation=True, max_length=512)
    return ds["train"].map(tok_fn, batched=True), ds["test"].map(tok_fn, batched=True)


# ────────────────────────────────────────────────
if __name__ == "__main__":
    ray.init()
    tmp_tok = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
    train_ds, eval_ds = get_dataset(tmp_tok)

    config = {
        "model_name":        "bigscience/bloom-3b",
        "per_device_batch":   1,
        "epochs":             2,
        "lr":                 2e-5,
        "grad_accum":         8,
        "ds_config_path":    "ds_zero3.json",
        "wandb_run":         "ray-bloom3b-zero3"
    }

    trainer = TransformersTrainer(
        trainer_init_per_worker = trainer_init_per_worker,
        trainer_init_config     = config,
        datasets                = {"train": train_ds, "evaluation": eval_ds},
        scaling_config          = ScalingConfig(num_workers=2, use_gpu=True),
        run_config              = RunConfig(
            name              = "llm_finetune_zero3",
            storage_path      = "./ray_results",
            checkpoint_config = CheckpointConfig(num_to_keep=2)
        ),
    )

    result = trainer.fit()
    print("Finished!  Wall-clock time (s):", result.metrics["time_total_s"])
