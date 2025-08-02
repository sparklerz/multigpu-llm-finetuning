# LLM Foundry FSDP Fine-tuning

## Objective
**Objective:** Fine-tune a [1.3B-parameter OPT model](https://huggingface.co/facebook/opt-1.3b) using MosaicML's LLM Foundry framework with PyTorch FSDP to enable efficient distributed training across multiple GPUs while leveraging pre-built, production-ready fine-tuning configurations.

## Technique & Tools
- **Technique & Tools:** Used MosaicML's LLM Foundry with PyTorch Fully Sharded Data Parallel (FSDP) for distributed training across 2 GPUs. LLM Foundry provides config-based fine-tuning with built-in FSDP sharding strategies, mixed precision training (amp_fp16), and activation checkpointing. Training code leverages Foundry's pre-built training loop with custom YAML configurations. Tracked experiment metrics with Weights & Biases (project: [llm-foundry-demo](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace), run: [opt_dolly_sft](https://wandb.ai/kannansarat9/llm-foundry-demo/runs/t1i96kr7)).

## Implementation Details
### LLM Foundry Configuration:
* **Framework:** MosaicML LLM Foundry v0.6.0 with config-based training setup
* **FSDP Strategy:** FULL_SHARD to distribute model parameters, gradients, and optimizer states across GPUs
* **Mixed Precision:** amp_fp16 for memory efficiency and faster training
* **Memory Optimization:** Activation checkpointing enabled, all-gather optimization with limit_all_gathers: true
* **Hardware:** 2 × T4 16GB GPUs on Kaggle Notebooks
* **Dataset:** Fine-tuned on [Dolly-15K instruction dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) with custom prompt/response formatting
* **Training Duration:** 2 epochs with cosine learning rate scheduler and warmup (85 batches)
* **Batch Configuration:** Global batch size of 16 with gradient accumulation of 4 steps
* **Checkpointing:** HuggingFace format checkpoints saved every epoch with automatic folder naming

## Results & Evaluation
### Training Performance:
* **Model Quality:** LLM Foundry's built-in evaluator demonstrated improved performance metrics with continued training, observable in the W&B metrics section
* **Memory Efficiency:** Successfully trained 1.3B parameter model on 2 × 16GB GPUs through FSDP parameter sharding
* **Training Stability:** Gradient clipping (norm=1.0) and cosine scheduler with warmup prevented training instabilities
* **Evaluation Interval:** Model evaluated every 100 batches for continuous monitoring

### Key Metrics:
* Foundry's inbuilt evaluation showed progressive improvement over training epochs
* Mixed precision training maintained numerical stability throughout training
* FSDP communication overhead minimized through optimized all-gather operations

## MLOps Practices
### Experiment Tracking
* **Experiment Tracking:** Comprehensive logging with Weights & Biases integration in project [llm-foundry-demo](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace) with run ID [opt_dolly_sft](https://wandb.ai/kannansarat9/llm-foundry-demo/runs/t1i96kr7). LLM Foundry automatically tracks training/validation loss, learning rate schedules, speed metrics, and model parameters. Built-in W&B callbacks provide real-time monitoring of training progress and evaluation results.

### Model Versioning
* **Model Versioning:** Multi-tier checkpoint strategy implemented:
* **Local Checkpoints:** HuggingFace format checkpoints saved in ./checkpoints with epoch-based folder structure (ep1, ep2)
* **Branch Versioning:** Final model weights saved to separate repository branches for each epoch:
  * [epoch-1 branch]()
  * [epoch-2 branch]()
* **Separate Repositories:** Individual HuggingFace repositories for each training epoch:
  * [llm-foundry-fsdp-opt-1.3B-epoch-1]()
  * [llm-foundry-fsdp-opt-1.3B-epoch-2]()

## How to Run
### Prerequisites
- **Platform:** Kaggle Notebooks with GPU acceleration (2 × T4 recommended)
- **Python 3.8+** with LLM Foundry dependencies
- **Hugging Face Account** with write access token
- **Weights & Biases Account** for experiment tracking

### Setup Instructions

