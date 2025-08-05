# Multi‑GPU LLM Fine‑Tuning Projects

This repository showcases hands-on projects leveraging multi-GPU training to fine-tune large language models (LLMs), demonstrating expertise in PyTorch Distributed, DeepSpeed, Ray (Tune, Train), and MosaicML's LLM Foundry. Each project includes detailed experiment tracking, evaluation, and final model weights.

## Projects Overview

| Project | Framework / Tool | Model | Hardware | Experiment Tracking | Resources |
|---------|------------------|-------|----------|---------------------|-----------|
| [PyTorch DDP Multi-GPU Training](./pytorch-ddp/) | PyTorch DDP | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | 2×T4 16GB | MLflow | 

<ul><li><a href="https://www.kaggle.com/code/saratkannan/pytorch-ddp-qwen-2gpus">Training Notebook</a></li><li><a href="https://www.kaggle.com/code/saratkannan/ddp-qwen-evaluation-notebook">Evaluation Notebook</a></li><li><a href="https://huggingface.co/ash001/pytorch-DDP-Qwen-0.5B/tree/main">HF Model</a></li></ul> |
| [PyTorch FSDP Multi-GPU Training](./pytorch-fsdp/) | PyTorch FSDP | [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b) | 2×T4 16GB | W&B | [Training Notebook](https://www.kaggle.com/code/saratkannan/pytorch-fsdp-opt-1-3b)<br>[Evaluation Notebook](https://www.kaggle.com/code/saratkannan/fsdp-opt-model-evaluation)<br>[HF Model](https://huggingface.co/ash001/pytorch-fsdp-opt-1.3B/tree/main)<br>[W&B](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv) |
| [DeepSpeed ZeRO-2 Offload Training](./deepspeed-offload/) | DeepSpeed ZeRO-2 Offload | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1×P100 16GB[^ds_offload] | W&B | [Training Notebook](https://www.kaggle.com/code/saratkannan/deepspeed-offload-llama-notebook)<br>[Evaluation Notebook](https://www.kaggle.com/code/saratkannan/deepspeed-offload-model-evaluation)<br>[HF Model](https://huggingface.co/ash001/deepspeed-offload-llama-3.2-1B/tree/main)<br>[W&B](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune) |
| [DeepSpeed Pipeline Parallelism](./deepspeed-pipeline/) | DeepSpeed Pipeline + ZeRO-1 | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 2×T4 16GB | W&B | [Training Notebook](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-notebook)<br>[Evaluation Notebook](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-evaluation-notebook)<br>[HF Model](https://huggingface.co/ash001/deepspeed-pipeline-llama-1B/tree/main)<br>[W&B](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline) |
| [LLM Foundry FSDP Fine-tuning](./llm-foundry-finetune/) | MosaicML's LLM Foundry, FSDP | [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b) | 2×T4 16GB | W&B | [Training Notebook](https://www.kaggle.com/code/saratkannan/llm-foundry-notebook)<br>[HF Model](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B/tree/epoch-2)<br>[W&B](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace) |
| [Ray Train with DeepSpeed ZeRO-3](./ray-train/) | Ray Train, DeepSpeed ZeRO-3 | [BLOOMZ-1b1](https://huggingface.co/bigscience/bloomz-1b1) | 2×T4 16GB | W&B | [Training Notebook](https://www.kaggle.com/code/saratkannan/ray-train-bloom-1b-notebook-start)<br>[HF Model](https://huggingface.co/ash001/ray-train-zero-3-bloom-1B/tree/main)<br>[W&B](https://wandb.ai/kannansarat9/ray-bloom-1b-zero3/workspace) |
| [Ray Tune Hyperparameter Optimization](./ray-tune/) | Ray Tune, PyTorch | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | 2×T4 16GB | W&B | [Training Notebook](https://www.kaggle.com/code/saratkannan/ray-tune-qwen-0-5b-notebook-6-trials)<br>[HF Model](https://huggingface.co/ash001/ray-tune-qwen-0.5B/tree/main)<br>[W&B](https://wandb.ai/kannansarat9/ray-tune-qwen/workspace) |

Most experiments were run on Kaggle with 2 × T4 16GB GPUs<br>
[^ds_offload]: DeepSpeed ZeRO-2 offload peaked at ~37 GB CPU RAM, exceeding Kaggle’s 30 GB CPU RAM limit, so the project was run on Vast.ai.
