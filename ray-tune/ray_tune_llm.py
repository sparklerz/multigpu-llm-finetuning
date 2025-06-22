import os, math, torch, ray, wandb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.air import session
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
    ds = load_dataset("imdb", split="train[:2%]")
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
        output_dir=trial_dir,
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
        save_strategy="epoch",                  # checkpoints handled by Ray
        logging_strategy="steps",
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(), # bf16 on ampere+
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    try:
        # ── run & report one epoch at a time ───────────────────────
        for epoch in range(int(config["epochs"])):
            trainer.train(resume_from_checkpoint=None)
            metrics = trainer.evaluate()

            # log to W&B
            wandb.log(
                {"eval_loss": metrics["eval_loss"], "epoch": epoch + 1}
            )
            trainer.save_model(trial_dir)
            # report to Ray Tune (drives ASHA)
            session.report(
                {"eval_loss": metrics["eval_loss"],
                 "training_iteration": epoch + 1},
                checkpoint=Checkpoint.from_directory(trial_dir)
            )
    finally:
        wandb.finish()

# 3 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ray.init()

    # ── Hyper-parameter search space ────────────────────────────────
    param_space = {
        "lr": tune.loguniform(1e-5, 5e-4),
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
            num_samples=8,                   # total trials; tweak as needed
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    best = results.get_best_result(metric="eval_loss", mode="min")
    print("\n🎯  Best hyper-params:", best.config)
    print("📉  Best eval-loss   :", best.metrics['eval_loss'])

    # ──  Push the best checkpoint to the Hub  ─────────────────────

    # Ray AIR checkpoint → local directory with model files
    best_dir = best.checkpoint.to_directory("best_checkpoint")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(best_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=True, trust_remote_code=True)

    repo_id = "ash001/ray-tune-qwen-0.5B"
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"✓ Pushed best model to https://huggingface.co/{repo_id}")