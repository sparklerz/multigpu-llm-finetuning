# Multi‑GPU LLM Fine‑Tuning Projects

This repository showcases practical projects leveraging multi-GPU training to fine-tune large language models (LLMs), demonstrating expertise in PyTorch Distributed, DeepSpeed, Ray (Tune, Train), and MosaicML's LLM Foundry. Each project includes detailed experiment tracking, evaluation, and final model weights.

## Projects Overview

| Project | Framework/Tool | Model | Hardware | Experiment Tracking | Resources |
|---------|----------------|-------|----------|---------------------|-----------|
| [PyTorch DDP Multi-GPU Training](pytorch-ddp-qwen-0.5b/) | PyTorch DDP | Qwen-0.5B |      | MLflow | [Notebooks](pytorch-ddp-qwen-0.5b/) |
| [PyTorch FSDP Distributed Training](pytorch-fsdp-opt-1.3b/) | PyTorch FSDP | OPT-1.3B |      | Wandb | [Notebook](pytorch-fsdp-opt-1.3b/pytorch-fsdp-opt-1-3b.ipynb) |
| [DeepSpeed ZeRO-2 Offload Training](deepspeed-zero2-offload-llama-1b/) | DeepSpeed Zero-2 Offload | LLaMA-1B |      | Wandb | [Notebook](deepspeed-zero2-offload-llama-1b/) |
| [DeepSpeed Pipeline Parallelism](deepspeed-pipeline-llama-1b/) | DeepSpeed Pipeline + Zero-1 | LLaMA-1B |      | Wandb | [Notebook](deepspeed-pipeline-llama-1b/deepspeed-pipeline-notebook.ipynb) |
| [LLM Foundry FSDP Fine-tuning](llm-foundry-opt-1.3b-fsdp/) | MosaicML's LLM Foundry, FSDP | OPT-1.3B |      | Wandb | [Notebook](llm-foundry-opt-1.3b-fsdp/llm-foundry-notebook.ipynb) |
| [Ray Train Distributed Training](ray-train-bloom-1b-zero3/) | Ray Train, DeepSpeed Zero-3 | BLOOMZ-1b1 |      | Wandb | [Notebooks](ray-train-bloom-1b-zero3/) |      |
| [Ray Tune Hyperparameter Optimization](ray-tune-qwen/) | Ray Tune, PyTorch | Qwen-0.5B |      | Wandb | [Notebook](ray-tune-qwen/ray-tune-qwen-0.5B-notebook-6-trials.ipynb), [HF Model](https://huggingface.co/ash001/ray-tune-qwen-0.5B) |



