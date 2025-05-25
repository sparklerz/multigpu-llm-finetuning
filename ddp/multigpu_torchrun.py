import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from huggingface_hub import hf_hub_download, HfApi


def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size()

class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        start_idx,
        end_idx,
        hf_repo,
        resume_file
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.hf_repo = hf_repo

        # total steps for this run
        total_samples = end_idx - start_idx
        world_size = dist.get_world_size()
        bs_per_gpu = self.dataloader.batch_size
        self.max_steps = (total_samples + bs_per_gpu * world_size - 1) // (bs_per_gpu * world_size)

        # resume if checkpoint provided
        self.epochs_run = 0
        self.global_step = 0
        if hf_repo and resume_file:
            try:
                path = hf_hub_download(repo_id=hf_repo, filename=resume_file)
            except:
                path = resume_file
            if os.path.exists(path) and self.local_rank == 0:
                ckpt = torch.load(path, map_location=self.device)
                state = ckpt.get("MODEL_STATE") or ckpt
                self.model.load_state_dict(state)
                self.epochs_run = ckpt.get("EPOCHS_RUN", 0)
                self.global_step = ckpt.get("GLOBAL_STEP", 0)
                print(f"[Rank {self.local_rank}] Resumed at step {self.global_step}, epoch {self.epochs_run}")

        # track samples processed per GPU
        self.processed_samples = self.global_step * bs_per_gpu

        # wrap model
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _save_checkpoint(self, epoch):
        if self.local_rank != 0:
            return
        # display epoch starting at 1
        disp_epoch = epoch + 1
        name = f"qwen2_0.5B_{self.start_idx}-{self.end_idx}-epoch-{disp_epoch}.pt"
        torch.save({
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": disp_epoch,
            "GLOBAL_STEP": self.global_step
        }, name)
        print(f"[Rank {self.local_rank}] Saved {name}")
        if self.hf_repo:
            HfApi().upload_file(
                path_or_fileobj=name,
                path_in_repo=name,
                repo_id=self.hf_repo,
                repo_type="model",
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            print(f"[Rank {self.local_rank}] Uploaded {name} to {self.hf_repo}")

    def train(self, num_epochs):
        for epoch in range(self.epochs_run, num_epochs):
            self.dataloader.sampler.set_epoch(epoch)
            for batch in self.dataloader:
                if self.global_step >= self.max_steps:
                    self._save_checkpoint(epoch)
                    return
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)

                # forward & backward
                self.optimizer.zero_grad()
                loss = self.model(**inputs, labels=labels).loss
                loss.backward()
                self.optimizer.step()

                # update counts
                self.global_step += 1
                self.processed_samples += labels.size(0)

                # dynamic log per batch
                print(f"[Rank {self.local_rank}] Processed {self.processed_samples} samples so far (step {self.global_step})")

                if self.global_step >= self.max_steps:
                    self._save_checkpoint(epoch)
                    return

            # end epoch checkpoint
            self._save_checkpoint(epoch)


def main(num_epochs: int,
         start_idx: int,
         end_idx: int,
         batch_size: int,
         hf_repo: str,
         resume_file: str = None):
    local_rank, world_size = ddp_setup()

    # MLflow
    mlflow.set_experiment("qwen2-0.5B-arxiv-finetune")
    mlflow.start_run(run_name=f"{world_size}gpu_{start_idx}-{end_idx}")
    mlflow.log_params({
        "num_epochs": num_epochs,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "batch_size": batch_size,
        "hf_repo": hf_repo
    })

    # load and slice dataset
    ds = load_dataset("ash001/arxiv-abstract", split="train")
    ds = ds.select(range(start_idx, end_idx))

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    def tok(ex):
        t = tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")
        t['labels'] = t['input_ids'].copy()
        return t

    tok_ds = ds.map(tok, remove_columns=ds.column_names)
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler = DistributedSampler(tok_ds, shuffle=True)
    loader = DataLoader(tok_ds, batch_size=batch_size, sampler=sampler,
                        collate_fn=collator, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(model, loader, optimizer,
                      start_idx, end_idx,
                      hf_repo, resume_file)
    trainer.train(num_epochs)

    # push tokenizer
    if local_rank == 0 and hf_repo:
        tokenizer.push_to_hub(hf_repo)

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DDP fine-tune Qwen2-0.5B by sample ranges with GPU sample logging")
    p.add_argument("num_epochs", type=int)
    p.add_argument("start_idx", type=int, help="Sample start index for this run")
    p.add_argument("end_idx", type=int, help="Sample end index for this run")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hf_repo", type=str, required=True,
                   help="HF repo ID, e.g. ash001/pytorch-DDP-Qwen2-0.5B")
    p.add_argument("--resume_file", type=str,
                   help="Checkpoint filename in repo, e.g. qwen2_0.5B_0-400000-epoch-1.pt")
    args = p.parse_args()
    main(args.num_epochs, args.start_idx, args.end_idx,
         args.batch_size, args.hf_repo, args.resume_file)
