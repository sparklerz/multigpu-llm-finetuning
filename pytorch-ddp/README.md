# PyTorch DDP Multi-GPU Training

## Objective
**Objective:** Compare single-GPU vs multi-GPU training performance using PyTorch's DistributedDataParallel (DDP) to fine-tune [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on arxiv abstracts dataset.

## Technique & Tools
* **Technique:** PyTorch DistributedDataParallel (DDP) with torchrun launcher for multi-GPU coordination. Implemented gradient accumulation and distributed sampling for efficient training across GPUs.
* **Tools:** PyTorch DDP, MLflow for experiment tracking and performance comparison, Hugging Face Transformers and Hub for model management, and custom dataset slicing for controlled experiments.

## Implementation Details
* **Multi-GPU Setup:** Used DDP with NCCL backend for efficient GPU communication
* **Dataset Slicing:** Trained on different data ranges (0-5000, 5000-15000, 15000-25000 samples) to enable fair performance comparisons
* **Gradient Accumulation:** Implemented configurable accumulation steps to handle memory constraints
* **Checkpointing:** Automatic checkpoint saving with semantic naming (data range + epoch indicators)
* **Hardware:** 2 × T4 16GB GPUs

## Results & Evaluation
### Performance Comparison:
* **2 GPU Speedup:** Observed **1.3× speedup** with 2 GPUs compared to single GPU training
* **Training Scale:** Comparative analysis performed on 10,000 samples using MLflow tracking
* **Model Quality:** Achieved similar evaluation loss on test data as the original pre-trained model (no overfitting observed)
* **Inference:** Model inference capabilities remain unchanged from base model

## Hardware & Compute
- **Platform:** Kaggle Notebooks
- **GPUs:** 2× NVIDIA T4 16GB 
- **Communication Backend:** NCCL for inter-GPU communication
- **Framework:** PyTorch DDP for distributed training

## MLOps Practices
### Experiment Tracking
- **MLflow Integration:** Comprehensive experiment tracking with separate runs for 1-GPU vs 2-GPU comparisons
- **Metrics Logged:** Training loss, total runtime, hyperparameters, and sample processing rates
- **Local Storage:** MLflow artifacts stored locally for analysis

### Model Versioning
- **Checkpoint Naming:** Semantic versioning with data range and epoch: `qwen2_0.5B_{start_idx}-{end_idx}-epoch-{epoch_num}.pt`
- **Hugging Face Hub:** Final model weights uploaded to [ash001/pytorch-DDP-Qwen2-0.5B](https://huggingface.co/ash001/pytorch-DDP-Qwen2-0.5B/tree/main)
- **Progressive Checkpointing:** Saves checkpoints after each epoch with resume capability

## How to Run
```
pip install
```









