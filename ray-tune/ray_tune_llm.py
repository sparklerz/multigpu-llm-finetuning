import os, math, torch, ray, wandb, pathlib, shutil
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.air import session, RunConfig, CheckpointConfig
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)

os.environ.setdefault("WANDB_PROJECT", "ray-tune-qwen")
os.environ.setdefault("WANDB_WATCH",   "false")          # skip parameter histograms

MODEL = "Qwen/Qwen2-0.5B-Instruct"

# 1 ──────────────────────────────────────────────────────────────────
# Helper: load + tokenise IMDb once per trial
def get_imdb(tokenizer):
    ds = load_dataset("imdb", split="train[:15%]")
    def tok_fn(ex):                     # truncate to fit GPU memory
        return tokenizer(ex["text"], truncation=True, max_length=256)
    tok = ds.map(tok_fn, batched=True, remove_columns=["text", "label"])
    split = tok.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]

# 2 ──────────────────────────────────────────────────────────────────
# Trial trainable
def train_fn(config):
    """Single Ray Tune trial → fine-tunes the model + reports val-loss."""
    # Reproducibility
    set_seed(config.get("seed", 42))

    model_name = config.get(
        "model_name",
        MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds, val_ds = get_imdb(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    run_name = f"tune-{session.get_trial_id()}"
    trial_dir = session.get_trial_dir()

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=run_name,
        config=config,
        reinit=True,
    )

    args = TrainingArguments(
        output_dir="/tmp/ignore",
        run_name=run_name,
        report_to=["wandb"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=max(1, 32 // config["batch_size"]),
        gradient_checkpointing=True,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        num_train_epochs=1,
        warmup_steps=config["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=1,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_bf16_supported(), # bf16 on ampere+
        fp16=not torch.cuda.is_bf16_supported(),
        fp16_full_eval=False,
        dataloader_pin_memory=True
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    try:
        prev_dir = None
        # ── run & report one epoch at a time ───────────────────────
        for epoch in range(int(config["epochs"])):
            trainer.train(resume_from_checkpoint=None)
            metrics = trainer.evaluate()
            # report to Ray Tune (drives ASHA)
            session.report(
                {"eval_loss": metrics["eval_loss"],
                 "training_iteration": epoch + 1}
            )

            # log to W&B
            wandb.log(
                {"eval_loss": metrics["eval_loss"], "epoch": epoch + 1}
            )
            if train.get_context().get_world_rank() == 0:
                if prev_dir and os.path.exists(prev_dir):
                    shutil.rmtree(prev_dir, ignore_errors=True)
                ckpt_dir = os.path.join(trial_dir, "ckpt")
                model.save_pretrained(ckpt_dir, safe_serialization=True)
                tokenizer.save_pretrained(ckpt_dir)
                prev_dir = ckpt_dir

        session.report(metrics, checkpoint=Checkpoint.from_directory(prev_dir))
    finally:
        wandb.finish()

# 3 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ray.init()

    # ── Hyper-parameter search space ────────────────────────────────
    param_space = {
        "lr": tune.loguniform(5e-6, 5e-5),
        "batch_size": tune.choice([1, 2, 4]),
        "weight_decay": tune.uniform(0.0, 0.3),
        "epochs": tune.choice([1, 2, 3]),
        "warmup_steps": tune.choice([0, 100, 200]),
        "seed": 42,
        # Constant entries still live in config for transparency
        "model_name": MODEL,
    }

    # ── Early-stopping scheduler (ASHA) ─────────────────────────────
    scheduler = ASHAScheduler(
        time_attr="training_iteration",      # = epoch here
        metric="eval_loss",
        mode="min",
        max_t=3,                             # epochs
        grace_period=1,
        reduction_factor=2,
    )

    # ── Build the tuner ─────────────────────────────────────────────
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn,
            resources={"cpu": 2, "gpu": 1}  # → 2 concurrent trials on 2 GPUs
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=4,                   # total trials
        ),
        param_space=param_space,
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="eval_loss", mode="min")
    for r in results:
        if r.path != best.path:
            shutil.rmtree(r.path, ignore_errors=True)
    print("\n🎯  Best hyper-params:", best.config)
    print("📉  Best eval-loss   :", best.metrics['eval_loss'])

    # ──  Push the best checkpoint to the Hub  ─────────────────────

    if best.checkpoint:
        with best.checkpoint.as_directory() as ckpt_dir:
            best_dir = pathlib.Path(ckpt_dir)
    else:
        best_dir = pathlib.Path(best.path) / "model"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(best_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=True, trust_remote_code=True)

    repo_id = "ash001/ray-tune-qwen-0.5B"
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"✓ Pushed best model to https://huggingface.co/{repo_id}")