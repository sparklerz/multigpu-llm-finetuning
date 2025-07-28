# PyTorch FSDP Multi-GPU Training

## Objective
**Objective:** Fine-tune a [1.3B-parameter OPT model](https://huggingface.co/facebook/opt-1.3b) using PyTorch's Fully Sharded Data Parallel (FSDP) to enable efficient training across multiple GPUs while managing memory constraints on consumer hardware.

## Technique & Tools
* **Technique:** PyTorch Fully Sharded Data Parallel (FSDP) with transformer auto-wrap policy, mixed precision training (FP16), and gradient checkpointing for memory optimization. Used `FULL_SHARD` strategy to distribute model parameters, gradients, and optimizer states across GPUs.
* **Tools:** PyTorch FSDP, Weights & Biases for experiment tracking (project: [opt-1.3B-fsdp-arxiv](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv)), Hugging Face Transformers and Hub for model management, and distributed sampling with NCCL backend for inter-GPU communication.

## Implementation Details
### FSDP Configuration:
* **Sharding Strategy:** `FULL_SHARD` to distribute all model components across GPUs
* **Auto-wrap Policy:** Transformer-based wrapping targeting `OPTDecoderLayer` modules
* **Mixed Precision:** FP16 for parameters, gradients, and buffers to reduce memory footprint
* **Memory Optimization:** Gradient checkpointing and disabled model caching
* **Hardware:** 2 × T4 16GB GPUs on Kaggle Notebooks
* **Dataset:** Fine-tuned on [arxiv abstracts](https://huggingface.co/datasets/ash001/arxiv-abstract) with configurable data slicing (start_idx to end_idx ranges)
* **Gradient Accumulation:** Configurable accumulation steps with gradient clipping (max_norm=1.0)

## Results & Evaluation
### Training Performance:
* **Model Quality:** Achieved similar evaluation loss on test data as the original pre-trained model (no overfitting observed)
* **Memory Efficiency:** Successfully trained 1.3B parameter model on 2 × 16GB GPUs through FSDP parameter sharding
* **Inference:** Model inference capabilities remain unchanged from base model
* **Scalability:** FSDP enabled training of models that wouldn't fit on single GPU memory

### Training Metrics:
- Gradient clipping successfully prevented gradient explosion
- Mixed precision training maintained numerical stability
- Distributed training completed without communication errors

## MLOps Practices
### Experiment Tracking
* **Weights & Biases Integration:** Comprehensive experiment logging with project [opt-1.3B-fsdp-arxiv](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv) and run ID [fsdp-opt-k76etxpy](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv/runs/k76etxpy). Tracked training loss, model parameters, gradients, and hyperparameters with automatic artifact uploading for checkpoints.
* **Monitoring:** Real-time loss tracking, gradient monitoring, and parameter watching through W&B dashboard for training insights and debugging.

### Model Versioning
* **Checkpoint Strategy:** Semantic checkpoint naming with data range and epoch indicators: `opt_1.3B_{start_idx}-{end_idx}-epoch-{epoch}.pt`
* **Hugging Face Integration:** Automatic upload of checkpoints to [ash001/pytorch-fsdp-opt-1.3B](https://huggingface.co/ash001/pytorch-fsdp-opt-1.3B/tree/main) with full state dict preservation for model sharing and deployment.
* **Resume Capability:** Built-in checkpoint resuming with global step and epoch tracking for interrupted training recovery.

## How to Run
### Prerequisites
- **Platform:** Kaggle Notebooks with GPU acceleration (2 × T4 recommended)
- **Python 3.8+** with PyTorch 2.0+ supporting FSDP
- **Hugging Face Account** with write access token
- **Weights & Biases** Account for experiment tracking

### Setup Instructions
#### 1. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.git
%cd multigpu-llm-finetuning
%cd pytorch-fsdp
```
#### 2. Install Dependencies
```
!pip install -r requirements.txt
```
#### 3. Authenticate with Hugging Face
```
from huggingface_hub import login
login(token="your_hf_token_here") # Replace with your actual write token
```
#### 4. Authenticate with Weights & Biases
```
!wandb login your_wandb_api_key_here  # Replace with your actual API key
```
**Note:** Replace with your actual W&B API key. You can find this in your W&B account settings.

## Training Configurations
### Multi-GPU FSDP Training
```
!torchrun --standalone --nproc_per_node=gpu fsdp_finetune.py \
  --num_epochs 1 \
  --start_idx 0 \
  --end_idx 5000 \
  --batch_size 4 \
  --accum_steps 16 \
  --initial_epoch 0 \
  --hf_repo ash001/pytorch-fsdp-opt-1.3B
```

### Resume from Checkpoint (if needed)
```
!torchrun --standalone --nproc_per_node=gpu fsdp_finetune.py \
  --num_epochs 1 \
  --start_idx 5000 \
  --end_idx 10000 \
  --batch_size 4 \
  --accum_steps 16 \
  --initial_epoch 0 \
  --hf_repo ash001/pytorch-fsdp-opt-1.3B \
  --resume_file opt_1.3B_0-5000-epoch-1.pt
```

**Note:** All training metrics will be automatically logged to your [W&B project dashboard](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv) for real-time monitoring.

### Parameter Explanation

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--num_epochs` | Number of training epochs | `1` |
| `--start_idx` | Data slice start index | `0` |
| `--end_idx` | Data slice end index | `5000` |
| `--batch_size` | Micro-batch size per GPU | `4` |
| `--accum_steps` | Gradient accumulation steps | `16` |
| `--initial_epoch` | Starting epoch for resume | `0` |
| `--hf_repo` | Hugging Face repository for uploads | Your repo name |
| `--resume_file` | Checkpoint file to resume from | Optional |

### Expected Outputs
- **Checkpoints:** Saved locally and uploaded to HF Hub automatically
- **W&B Tracking:** Real-time experiment tracking in [opt-1.3B-fsdp-arxiv](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv) project
- **Console Output:** Detailed training progress with step-by-step loss reporting
- **Model Artifacts:** Final model weights available on Hugging Face Hub

### Performance Notes
- **Memory Usage:** FSDP efficiently distributes 1.3B parameters across 2 × 16GB GPUs
- **Effective Batch Size:** 128 (4 per-GPU batch × 16 accumulation steps × 2 GPUs)
- **Communication:** NCCL backend ensures efficient inter-GPU communication
- **Training Speed:** Gradient accumulation and mixed precision optimize training throughput

### Model Evaluation
Use the provided evaluation notebook to assess model performance:
- **Evaluation Notebook:** [fsdp-opt-model-evaluation.ipynb](https://www.kaggle.com/code/saratkannan/fsdp-opt-model-evaluation) - Comprehensive loss comparison and text generation quality assessment

**Key Findings:**
- **Loss Preservation:** Test data evaluation loss matches original pre-trained model performance
- **No Overfitting:** Model maintains generalization capabilities after fine-tuning
- **Inference Quality:** Generated text quality comparable to base model

## Links & References

### Notebooks
- **Training Notebook:** [pytorch-fsdp-opt-1-3b.ipynb](https://www.kaggle.com/code/saratkannan/pytorch-fsdp-opt-1-3b) - Complete FSDP training workflow and setup
- **Evaluation Notebook:** [fsdp-opt-model-evaluation.ipynb](https://www.kaggle.com/code/saratkannan/fsdp-opt-model-evaluation) - Model performance assessment and loss analysis

### Model Artifacts
- **Fine-tuned Weights:** [ash001/pytorch-fsdp-opt-1.3B](https://huggingface.co/ash001/pytorch-fsdp-opt-1.3B/tree/main) - All training checkpoints and final model
- **Base Model:** [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b) - Original OPT-1.3B model

### Experiment Tracking
- **W&B Project:** [opt-1.3B-fsdp-arxiv](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv) - Complete training metrics and artifacts
- **Run ID:** [fsdp-opt-k76etxpy](https://wandb.ai/kannansarat9/opt-1.3B-fsdp-arxiv/runs/k76etxpy) - Specific experiment run with detailed logs and model monitoring
