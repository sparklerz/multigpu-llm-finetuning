# PyTorch DDP Multi-GPU Training

## Objective
**Objective:** Compare single-GPU vs multi-GPU training performance using PyTorch's DistributedDataParallel (DDP) to fine-tune [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on arxiv abstracts dataset.

## Technique & Tools
* **Technique:** PyTorch DistributedDataParallel (DDP) with torchrun launcher for multi-GPU coordination. Implemented gradient accumulation and distributed sampling for efficient training across GPUs.
* **Tools:** PyTorch DDP, MLflow for experiment tracking and performance comparison, Hugging Face Transformers and Hub for model management, and custom dataset slicing for controlled experiments.

## Implementation Details
* **Multi-GPU Setup:** Used DDP with NCCL backend for efficient GPU communication
* **Dataset Slicing:** Trained on different data ranges (0-5000, 5000-10000 samples) to enable fair performance comparisons
* **Gradient Accumulation:** Implemented configurable accumulation steps to handle memory constraints
* **Checkpointing:** Automatic checkpoint saving with semantic naming (data range + epoch indicators)
* **Hardware:** 2 × T4 16GB GPUs

## Results & Evaluation
### Performance Comparison:
* **2 GPU Speedup:** Observed **1.34× speedup** with 2 GPUs compared to single GPU training
* **Training Scale:** Comparative analysis performed on 5,000 samples using MLflow tracking
* **Model Quality:** Achieved similar evaluation loss on test data as the original pre-trained model (no overfitting observed)
* **Inference:** Model inference capabilities remain unchanged from base model

## Hardware & Compute
- **Platform:** Kaggle Notebooks
- **Alternative Platform:** Vast.ai with open port access for real-time MLflow monitoring
- **GPUs:** 2× NVIDIA T4 16GB 
- **Communication Backend:** NCCL for inter-GPU communication
- **Framework:** PyTorch DDP for distributed training

## MLOps Practices
### Experiment Tracking
- **MLflow Integration:** Comprehensive experiment tracking for performance comparison between GPU configurations
- **Real-time Monitoring:** Optional Vast.ai deployment with web UI (port 8082) for live dashboard access
- **Kaggle Compatible:** Full MLflow functionality available in Kaggle notebooks with post-training analysis
- **Metrics Logged:** Training loss curves, training phase runtime and hyperparameters
- **Local Access:** MLflow UI accessible locally via `mlflow ui --backend-store-uri ./` on port 5000


### Model Versioning
- **Checkpoint Naming:** Semantic versioning with data range and epoch: `qwen2_0.5B_{start_idx}-{end_idx}-epoch-{epoch_num}.pt`
- **Hugging Face Hub:** Final model weights uploaded to [ash001/pytorch-DDP-Qwen-0.5B](https://huggingface.co/ash001/pytorch-DDP-Qwen-0.5B/tree/main)
- **Progressive Checkpointing:** Saves checkpoints after each epoch with resume capability

## How to Run
### Prerequisites
- **Primary Platform:** Kaggle Notebooks with GPU acceleration enabled (2 × T4 recommended)
- **Optional Platform:** Vast.ai for real-time MLflow monitoring (paid service)
- **Hugging Face Account** with write access token
- **Git** and **Python 3.8+**

### MLflow Experiment Tracking Setup

#### Option 1: Vast.ai (Optional - Real-time Monitoring)
#### 1. Install MLflow
```
pip install mlflow
```
#### 2. Start MLflow UI (accessible during training)
```
mlflow ui --host 0.0.0.0 --port 8082
```
**Note:** Ensure port 8082 is open on your Vast.ai instance for external access.

**Post-Training Access:** Use "Upload/Download Data from Cloud Providers" (cloud sync) feature to download experiment zip file

**Benefits:** Real-time experiment monitoring, live dashboard access

#### Option 2: Kaggle Notebooks (Primary Platform)
**Default Setup:** MLflow runs automatically in Kaggle notebooks - no additional configuration needed.
**Post-Training Access:** Download zip file directly from notebook outputs for local analysis.

#### Option 3: Local Analysis (Post-training)

#### 1. Extract and navigate to MLflow artifacts directory
```
unzip mlruns-DDP-two-GPUs-0-5000.zip # or mlruns-DDP-one-GPU-5000-10000.zip
cd mlruns-DDP-two-GPUs-0-5000/mlruns # or mlruns-DDP-one-GPU-5000-10000/mlruns
```
#### 2. Start local MLflow UI
```
mlflow ui --backend-store-uri ./ --host 127.0.0.1 --port 5000
```
#### 3. Visit: http://127.0.0.1:5000/

**Use Case:** Offline analysis of completed experiments


### Setup Instructions
#### 1. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.git
%cd multigpu-llm-finetuning
%cd pytorch-ddp
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

## Training Configurations
### Multi-GPU Training (2 GPUs)
```
!torchrun --standalone --nproc_per_node=2 multigpu_torchrun.py \
  --num_epochs 1 \
  --start_idx 0 \
  --end_idx 5000 \
  --batch_size 1 \
  --accum_steps 8 \
  --hf_repo ash001/pytorch-DDP-Qwen-0.5B
```

### Single-GPU Training (with Resume)
```
!torchrun --standalone --nproc_per_node=1 multigpu_torchrun.py \
  --num_epochs 1 \
  --start_idx 5000 \
  --end_idx 10000 \
  --batch_size 1 \
  --accum_steps 8 \
  --hf_repo ash001/pytorch-DDP-Qwen-0.5B \
  --resume_file qwen2_0.5B_0-5000-epoch-1.pt
```

### Archive MLflow Results (After Training)
#### For 2-GPU experiments
```
!zip -r mlruns-DDP-two-GPUs-0-5000.zip mlruns
```

#### For single-GPU experiments
```
!zip -r mlruns-DDP-one-GPU-5000-10000.zip mlruns
```

**Note:** Create zip archives for easy download and sharing of experiment results.

### Parameter Explanation

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--num_epochs` | Number of epochs | `1` |
| `--start_idx` | Data slice range start index | `0` |
| `--end_idx` | Data slice range end index | `5000` |
| `--batch_size` | Micro-batch size per GPU | `1` |
| `--accum_steps` | Gradient accumulation steps | `8` |
| `--hf_repo` | Hugging Face repository for uploads | Your repo name |
| `--resume_file` | Checkpoint file to resume from | Optional |

### Expected Outputs
- **Checkpoints:** Saved locally and uploaded to HF Hub
- **MLflow Logs:** Experiment tracking data stored locally
- **Console Output:** Training progress with loss metrics
- **Final Model:** Uploaded to specified Hugging Face repository

#### MLflow Results Access
- **During Training:** Live dashboard on Vast.ai MLflow UI
- **Post-Training:** Download experiment zip files (`mlruns-DDP-two-GPUs-0-5000.zip` or `mlruns-DDP-one-GPU-5000-10000.zip`) for local analysis
- **Key Metrics:** Training loss vs steps, total runtime per GPU configuration

#### Performance Notes
- **2-GPU Setup:** Expect ~1.34× speedup compared to single GPU
- **Memory Usage:** Each T4 GPU (16GB) should handle batch_size=1 comfortably
- **Training Time:** Varies based on data slice size and hardware

### Model Evaluation
Use the provided evaluation notebook to assess model performance: [Model Evaluation Notebook](https://www.kaggle.com/code/saratkannan/ddp-qwen-evaluation-notebook)

**Key Findings:**
- **Model Quality:** Achieved similar evaluation loss on test data as the original pre-trained model (no overfitting observed)
- **Inference:** Model inference capabilities remain unchanged from base model

## Experiment Results
### Training Performance Comparison

![Training Performance Analysis](https://github.com/user-attachments/assets/8b0635d8-a77c-4636-9af7-6491e4bbb739)
*Performance comparison showing 1.34× speedup with 2-GPU DDP vs single-GPU training*

### Key Findings
- **Speedup:** 1.34× improvement with 2-GPU DDP configuration
- **Loss Convergence:** Similar convergence patterns across GPU configurations
- **Efficiency:** Marginal speedup indicates communication overhead in DDP setup


## Links & References

### Notebooks:

- [Model Evaluation Notebook](https://www.kaggle.com/code/saratkannan/ddp-qwen-evaluation-notebook) - Comprehensive evaluation comparing loss and text generation quality across all fine-tuned checkpoints
- [Training Workflow Example](https://www.kaggle.com/code/saratkannan/pytorch-ddp-qwen-2gpus) - Complete DDP training setup and execution (0-5000 samples)

**Note:** All training phases follow the same workflow with different parameters as documented in the "How to Run" section.

### Model Artifacts:
- [Fine-tuned Model Weights](https://huggingface.co/ash001/pytorch-DDP-Qwen-0.5B/tree/main) - All training checkpoints and final model
- [Base Model Repository](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) - Original Qwen2-0.5B-Instruct model
