# Ray Train with DeepSpeed ZeRO-3

## Objective
**Objective:** Fine-tune a [1.1B-parameter BLOOMZ model](https://huggingface.co/bigscience/bloomz-1b1) using Ray Train with DeepSpeed ZeRO-3 for distributed training across multiple GPUs, demonstrating scalable LLM fine-tuning with minimal configuration overhead.

## Technique & Tools
- **Technique:** Ray Train with DeepSpeed ZeRO-3 for distributed training with full parameter sharding across GPUs. Used config-based minimal Ray code to orchestrate distributed training with automatic parameter partitioning and gradient synchronization.
- **Tools:** Ray Train for distributed orchestration, DeepSpeed ZeRO-3 for memory-efficient parameter sharding, PyTorch for training implementation, Weights & Biases for experiment tracking (project: [ray-bloom-1b-zero3](https://wandb.ai/kannansarat9/ray-bloom-1b-zero3/workspace)), and Hugging Face Hub for model management.

## Implementation Details
### Ray Train Configuration:
* **Scaling Config:** 2 workers with 1 GPU per worker (2 × T4 16GB total)
* **DeepSpeed ZeRO-3:** Full parameter sharding with `stage3_gather_16bit_weights_on_model_save` enabled
* **Memory Optimization:** FP16 training, gradient checkpointing, and paged AdamW 8-bit optimizer
* **Dataset:** Fine-tuned on [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) with custom tokenization (512 max length, 90/10 train/test split)
* **Training Parameters:** 1 epoch, learning rate 2e-5, batch size 1 per device, 8 gradient accumulation steps
* **Automatic Uploads:** Direct checkpoint uploading to Hugging Face Hub during training

### Key Configuration Features:
* Config-based approach with minimal Ray boilerplate code
* Automatic distributed setup with NCCL P2P disabled for stability
* Custom callbacks for wall-clock time tracking and mean loss computation
* Integrated W&B logging from rank 0 worker only

## Results & Evaluation
### Training Performance:
* **Model Quality:** Ray Train's built-in evaluator logged lesser loss than the initial training loss, demonstrating successful fine-tuning
* **Memory Efficiency:** Successfully trained 1.1B parameter model across 2 × 16GB GPUs using ZeRO-3 parameter sharding
* **Evaluation Metrics:** Evaluation loss reduction observable in W&B eval section, confirming model improvement over training

### Training Metrics:
* **Effective batch size:** 16 (1 per-device × 8 accumulation × 2 workers)
* Mixed precision training (FP16) maintained numerical stability
* Distributed training completed successfully with automatic checkpoint saving

## MLOps Practices
### Experiment Tracking
* **Experiment Tracking:** Logged training and evaluation metrics to Weights & Biases with project [ray-bloom-1b-zero3](https://wandb.ai/kannansarat9/ray-bloom-1b-zero3/workspace) and run ID: [worker-0](https://wandb.ai/kannansarat9/ray-bloom-1b-zero3/runs/vk6bl79s). Tracked comprehensive training progress, loss curves, and system metrics with real-time monitoring capabilities.

### Model Versioning
* **Model Versioning:** No traditional model versioning implemented - fine-tuned weights trained across the complete dataset are directly uploaded to Hugging Face Hub ([ash001/ray-train-zero-3-bloom-1B](https://huggingface.co/ash001/ray-train-zero-3-bloom-1B/tree/main)) for immediate deployment and sharing.

## How to Run
### Prerequisites
- **Platform:** Kaggle Notebooks or similar environment with multi-GPU support (2 × T4 recommended)
- **Python 3.8+** with Ray 2.12.0 and DeepSpeed 0.14.5
- **Hugging Face Account** with write access token
- **Weights & Biases** Account for experiment tracking

### Setup Instructions
#### 1. Verify Python Version
```
!python --version
```
#### 2. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.gi
%cd multigpu-llm-finetuning
%cd ray-train
```
#### 3. Clean Previous PyTorch Installations
```
!pip uninstall -y peft accelerate torch torchvision torchaudio triton
```
**Note:** This step ensures clean installation by removing potentially conflicting packages

#### 4. Install Dependencies
```
!pip install -qU --no-cache-dir -r requirements.txt
```
#### 5. Install Git LFS Support
```
!apt-get -qq update
!apt-get -qq install git-lfs
!git lfs install
```
**Note:** Git LFS is required for handling large model files during upload/download

#### 6. Verify Package Versions
```
import transformers, peft, accelerate, deepspeed, os, torch, torchvision
print(f"[{os.getenv('RANK', 'driver')}] "
      f"torch {torch.__version__}, "
      f"torchvision {torchvision.__version__}, "
      f"transformers {transformers.__version__}, "
      f"peft {peft.__version__}, "
      f"accelerate {accelerate.__version__}, "
      f"deepspeed {deepspeed.__version__}")
```
#### 7. Authenticate with Hugging Face
```
from huggingface_hub import login
login(token="your_hf_write_token_here")  # Replace with your actual write token
```
#### 8. Set Environment Variable for HF Token
```
%env HF_TOKEN=your_hf_write_token_here
```
**Note:** Replace `your_hf_write_token_here` with your actual HuggingFace write token.
#### 9. Authenticate with Weights & Biases
```
!wandb login your_wandb_api_key_here  # Replace with your actual API key
```
**Note:** Replace with your actual W&B API key. You can find this in your W&B account settings.

#### 10. Configure CUDA Memory Allocation
```
!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```
**Note:** This step optimizes CUDA memory allocation for multi-GPU training

#### 11. Run Fine-tuning
```
!python ray_train_llm.py
```

### Expected Training Flow
- **Ray Initialization:** Ray cluster setup with 2 workers
- **Model Loading:** BLOOMZ-1B1 model distributed across GPUs using ZeRO-3
- **Dataset Processing:** IMDB dataset tokenized and split automatically
- **Training Progress:** Real-time logging to W&B with loss tracking
- **Checkpoint Upload:** Automatic model upload to Hugging Face Hub
- **Completion:** Final metrics displayed including total training time

### Configuration
Edit the config dictionary in `ray_train_llm.py` to customize:
- **Model:** Change `model_name` to use different base models
- **Batch Size:** Adjust `per_device_batch` for memory constraints
- **Epochs:** Set `epochs` for training duration
- **Learning Rate:** Modify `lr` for different convergence behavior
- **DeepSpeed Config:** Edit `ds_zero3.json` for ZeRO-3 parameters

### Expected Outputs
- **Ray Results:** Training logs and checkpoints in `ray_results/` directory
- **W&B Tracking:** Real-time experiment monitoring in ray-bloom-1b-zero3 project
- **HF Hub Upload:** Automatic model checkpoint uploads during training
- **Console Output:** Detailed training progress with distributed worker coordination

## Links & References

### Notebooks
Training progression notebooks demonstrating the complete workflow:
- [ray-train-bloom-1b-notebook-start.ipynb]() - Initial setup and configuration
- [ray-train-bloom-1b-notebook-mid-1.ipynb]() through [ray-train-bloom-1b-notebook-mid-6.ipynb]() - Training progress checkpoints
- [ray-train-bloom-1b-notebook-end.ipynb]() - Final results and evaluation

### Model Artifacts
- **Base Model:** [bigscience/bloomz-1b1](https://huggingface.co/bigscience/bloomz-1b1) - Original BLOOMZ-1B1 model
- **Fine-tuned Weights:** [ash001/ray-train-zero-3-bloom-1B](https://huggingface.co/ash001/ray-train-zero-3-bloom-1B/tree/main) - Complete fine-tuned model with ZeRO-3 training

### Experiment Tracking
- **W&B Project:** [ray-bloom-1b-zero3](https://wandb.ai/kannansarat9/ray-bloom-1b-zero3/workspace) - Complete training metrics and evaluation results
- **Run ID:** [worker-0](https://wandb.ai/kannansarat9/ray-bloom-1b-zero3/runs/vk6bl79s) - Specific experiment run with distributed training logs and loss monitoring
