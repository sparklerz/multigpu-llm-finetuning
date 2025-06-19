import os, time, torch, ray
from datasets import load_dataset
from ray.train.torch import TorchTrainer
from ray.train.huggingface.transformers import (
    prepare_trainer, RayTrainReportCallback
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          TrainingArguments,
                          Trainer,
                          TrainerCallback)
from huggingface_hub import Repository
import json, pathlib
import wandb

HF_REPO = "ash001/ray-train-zero-3-bloom-1B"

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
# 1 Build a vanilla HF Trainer (will be wrapped by Ray)
def trainer_init_per_worker(train_dataset=None, eval_dataset=None, **cfg):
    model_name = cfg["model_name"]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tok   = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    tok.pad_token = tok.eos_token        # safety for causal models
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir           = "./outputs",
        eval_strategy        = "steps",
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
        deepspeed           = cfg["ds_config"],
        fp16                = True,
        optim               = "paged_adamw_8bit",
        report_to           = "wandb",
        run_name            = cfg["wandb_run"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[WallClockCallback(), HubTagEpochCallback()],
        tokenizer=tok,
    )
    # Ray glue
    trainer.add_callback(RayTrainReportCallback())
    return prepare_trainer(trainer)

# ────────────────────────────────────────────────
# Ray train-loop entry point
def train_loop_per_worker(cfg):
    if ray.train.get_context().get_world_rank() == 0:
        wandb.init(project="ray-bloom-1b-zero3",
                name=f"worker-{os.environ.get('RANK', '0')}",
                reinit=True)
    train_ds = cfg.pop("train_ds")
    eval_ds  = cfg.pop("eval_ds")
    trainer = trainer_init_per_worker(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        **cfg
    )
    trainer.train()


# ────────────────────────────────────────────────
# 2  Dataset: IMDB
def get_dataset(tok):
    ds = load_dataset("imdb", split="train[:20%]")
    def tok_fn(ex): return tok(ex["text"], truncation=True, max_length=512)
    tokenised = ds.map(tok_fn, batched=True, remove_columns=["text", "label"])
    split = tokenised.train_test_split(test_size=0.1)
    return split["train"], split["test"]


# ────────────────────────────────────────────────
if __name__ == "__main__":
    ray.init()
    ds_conf = json.load(open(pathlib.Path(__file__).parent / "ds_zero3.json"))

    config = {
        "model_name":        "bigscience/bloomz-1b1",
        "per_device_batch":   1,
        "epochs":             2,
        "lr":                 2e-5,
        "grad_accum":         8,
        "ds_config":          ds_conf,
        "wandb_run":         "ray-bloom-1b-zero3"
    }

    # Download & tokenise IMDb once on the driver
    tok_driver = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    train_ds, eval_ds = get_dataset(tok_driver)
    config["train_ds"] = train_ds
    config["eval_ds"]  = eval_ds

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config       = config,
        scaling_config          = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1}),
        run_config              = RunConfig(
            name              = "llm_finetune_zero3",
            storage_path      = f"file://{os.path.abspath('ray_results')}",
            checkpoint_config = CheckpointConfig(num_to_keep=2)
        ),
    )

    result = trainer.fit()
    print("Finished!  Wall-clock time (s):", result.metrics["time_total_s"])
