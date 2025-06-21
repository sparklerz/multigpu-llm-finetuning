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
from huggingface_hub import Repository, HfApi, login
import json, pathlib
import ray
import wandb

HF_REPO = "ash001/ray-train-zero-3-bloom-1B"
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN is not set."
    )
login(token=HF_TOKEN)

# ────────────────────────────────────────────────
# 0  Simple W&B time-tracking callback
class WallClockCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            self._start = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            wandb.log({"train/runtime_seconds": time.time() - self._start})

class MeanLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        mean_loss = logs["loss"] / args.gradient_accumulation_steps

        # keep the metric inside Hugging Face’s own log dict too
        logs["mean_loss"] = mean_loss

        # push to wandb – but only once per distributed step
        if (ray.train.get_context().get_world_rank() == 0 and wandb.run is not None):
            wandb.log({"train/mean_loss": mean_loss}, step=state.global_step)

class HubTagEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **_):
        if (torch.distributed.get_rank() == 0 and ray.train.get_context().get_world_rank() == 0):
            # create /<output_dir>/hf_repo once and clone into it
            repo_dir = os.path.join(args.output_dir, "hf_repo")
            os.makedirs(repo_dir, exist_ok=True)                # safe even if it already exists

            # if the folder isn't a git repo yet, clone; otherwise just reopen it
            if not os.path.isdir(os.path.join(repo_dir, ".git")):
                Repository(repo_dir, clone_from=HF_REPO, token=HF_TOKEN)
            repo = Repository(repo_dir, token=HF_TOKEN)

            tag = f"epoch-{int(state.epoch)}"
            repo.add_tag(tag)
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
        eval_strategy        = "epoch",
        per_device_train_batch_size = cfg["per_device_batch"],
        per_device_eval_batch_size  = cfg["per_device_batch"],
        learning_rate        = cfg["lr"],
        num_train_epochs     = cfg["epochs"],
        logging_steps        = 1,
        gradient_accumulation_steps = cfg["grad_accum"],
        gradient_checkpointing       = True,
        save_strategy        = "epoch",
        save_total_limit     = 1,
        save_on_each_node    = False,
        push_to_hub          = True,
        hub_model_id         = HF_REPO,
        hub_strategy         = "checkpoint",
        hub_token            = HF_TOKEN,
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
        callbacks=[WallClockCallback(), MeanLossCallback(), HubTagEpochCallback()],
        tokenizer=tok,
    )
    # Ray glue
    trainer.add_callback(RayTrainReportCallback())
    return prepare_trainer(trainer)

# ────────────────────────────────────────────────
# Ray train-loop entry point
def train_loop_per_worker(cfg):
    hf_token = cfg.pop("hf_token", None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
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
    ds = load_dataset("imdb", split="train[:1%]")
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
        "epochs":             1,
        "lr":                 2e-5,
        "grad_accum":         8,
        "ds_config":          ds_conf,
        "wandb_run":         "ray-bloom-1b-zero3",
        "hf_token":          HF_TOKEN
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
            checkpoint_config = CheckpointConfig(num_to_keep=1)
        ),
    )

    result = trainer.fit()
    print("Finished!  Wall-clock time (s):", result.metrics["time_total_s"])
