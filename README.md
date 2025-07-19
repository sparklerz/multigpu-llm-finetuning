# Multi‑GPU LLM Fine‑Tuning Projects

This repository showcases practical projects leveraging multi-GPU training to fine-tune large language models (LLMs), demonstrating expertise in PyTorch Distributed, DeepSpeed, Ray (Tune, Train), and MosaicML's LLM Foundry. Each project includes detailed experiment tracking, evaluation, and final model weights.

## Projects Overview

| Project | Framework / Tool | Model | Hardware | Experiment Tracking | Resources |
|---------|------------------|-------|----------|---------------------|-----------|
| [PyTorch DDP Multi-GPU Training](./pytorch-ddp/) | PyTorch DDP | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |      | MLflow | [Notebooks](pytorch-ddp-qwen-0.5b/) |
| [PyTorch FSDP Distributed Training](./pytorch-fsdp/) | PyTorch FSDP | [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b) |      | W&B | [Notebook](pytorch-fsdp-opt-1.3b/pytorch-fsdp-opt-1-3b.ipynb) |
| [DeepSpeed ZeRO-2 Offload Training](./deepspeed-offload/) | DeepSpeed Zero-2 Offload | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |      | W&B | [Notebook](deepspeed-zero2-offload-llama-1b/) |
| [DeepSpeed Pipeline Parallelism](./deepspeed-pipeline/) | DeepSpeed Pipeline + Zero-1 | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |      | W&B | [Notebook](deepspeed-pipeline-llama-1b/deepspeed-pipeline-notebook.ipynb) |
| [LLM Foundry FSDP Fine-tuning](./llm-foundry-finetune/) | MosaicML's LLM Foundry, FSDP | [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b) |      | W&B | [Notebook](llm-foundry-opt-1.3b-fsdp/llm-foundry-notebook.ipynb) |
| [Ray Train Distributed Training](./ray-train/) | Ray Train, DeepSpeed Zero-3 | [BLOOMZ-1b1](https://huggingface.co/bigscience/bloomz-1b1) |      | W&B | [Notebooks](ray-train-bloom-1b-zero3/) |      |
| [Ray Tune Hyperparameter Optimization](./ray-tune/) | Ray Tune, PyTorch | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |      | W&B | [Notebook](ray-tune-qwen/ray-tune-qwen-0.5B-notebook-6-trials.ipynb)<br>[HF Model](https://huggingface.co/ash001/ray-tune-qwen-0.5B) |



