# DeepSpeed ZeRO-2 Offload Training

## Objective
**Objective:** Fine-tune a [1B-parameter LLaMA model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on a single 16GB GPU by leveraging DeepSpeed's ZeRO-2 Offload to CPU memory, demonstrating efficient training of large language models on modest hardware configurations.

## Technique & Tools
- **Technique:** Used DeepSpeed ZeRO-2 Offload for optimizer state and gradient offloading to CPU memory, enabling training of billion-parameter models on single GPU setups. Training code written in PyTorch with distributed processing handled by DeepSpeed's engine and automatic mixed precision (FP16) for memory efficiency.
- **Tools:** DeepSpeed ZeRO Stage 2 with CPU offload, PyTorch for model implementation, DeepSpeedCPUAdam optimizer for CPU-offloaded optimization states, and Weights & Biases for experiment tracking (project: [deepspeed-llama-3.2-1B-finetune](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune), run: [clear-silence-11](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune/runs/8jways6j)).

## Implementation Details
### DeepSpeed Configuration:
* **ZeRO Stage:** Stage 2 with optimizer state offloading to CPU memory
* **Memory Optimization:** Offloaded optimizer states to host memory to handle GPU memory limitations
* **Mixed Precision:** FP16 enabled for reduced memory footprint and faster training
* **Hardware:** 1 × Tesla P100 16GB GPU with 37GB CPU RAM on Vast.ai instance (requires minimum 37GB CPU RAM - not possible on Kaggle due to memory constraints)
* **Dataset:** Fine-tuned on [arxiv abstracts](https://huggingface.co/datasets/ash001/arxiv-abstract) with 4,000 samples for 1 epoch
* **Checkpoint Strategy:** Saved periodic checkpoints with naming convention indicating data index range and epoch (`llama3.2-1B_{start_idx}-{end_idx}-epoch-{epoch}.pt`)

### Training Configuration:
* **Gradient Accumulation:** Configurable accumulation steps for effective larger batch sizes
* **Data Slicing:** Configurable start and end indices for dataset partitioning
* **Resume Capability:** Built-in checkpoint resuming from Hugging Face Hub with epoch tracking

## Results & Evaluation
### Training Performance:
* **Model Quality:** Achieved similar evaluation loss on test data as the original pre-trained model, demonstrating successful fine-tuning without overfitting
* **Memory Efficiency:** Successfully demonstrated that a 1B+ parameter model can be trained on a single Tesla P100 16GB GPU through CPU offloading
* **Inference:** Model inference capabilities remain unchanged from base model - no modifications to inference pipeline required

### Training Metrics:
* **Hardware Utilization:** Effectively utilized 16GB GPU memory with 37GB CPU RAM offloading for optimizer states
* **Scalability:** Proved viability of training large language models on consumer-grade hardware through memory offloading techniques

## MLOps Practices
### Experiment Tracking
* **Weights & Biases Integration:** Comprehensive experiment logging with project [deepspeed-llama-3.2-1B-finetune](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune) and run ID [clear-silence-11](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune/runs/8jways6j). Tracked training and evaluation loss, hyperparameters, and model performance metrics with real-time monitoring during training.
* **Monitoring:** Real-time loss tracking and training progress monitoring through W&B dashboard for training insights and performance analysis.

### Model Versioning
* **Checkpoint Strategy:** Semantic checkpoint naming with data range and epoch indicators: `llama3.2-1B_{start_idx}-{end_idx}-epoch-{epoch}.pt`
* **Hugging Face Integration:** Uploaded final model weights to Hugging Face Hub at [ash001/deepspeed-offload-llama-3.2-1B](https://huggingface.co/ash001/deepspeed-offload-llama-3.2-1B/tree/main). Checkpoints include indices to indicate training progress and data coverage.
* **Version Control:** Systematic versioning approach enabling reproducible training and easy model retrieval.

## How to Run
### Prerequisites
- **Platform:** Vast.ai instance with Tesla P100 16GB GPU and minimum 37GB CPU RAM for optimizer offloading
- **Python 3.8+** with DeepSpeed installation supporting CPU Adam optimizer
- **Weights & Biases** Account for experiment tracking
- **Hugging Face Account** with approved access to Llama models and both read/write tokens
- **Note:** You must request access to [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) before starting training


### Setup Instructions
#### 1. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.git
%cd multigpu-llm-finetuning
%cd deepspeed-offload
```
#### 2. Install Dependencies
```
!pip install -r requirements.txt
```
#### 3. Authenticate with Weights & Biases
```
!wandb login your_wandb_api_key_here  # Replace with your actual API key
```
**Note:** Replace with your actual W&B API key. You can find this in your W&B account settings.

#### 4. Authenticate with Hugging Face
**Read Token (for downloading Llama model - requires approved access)**
```
from huggingface_hub import login
login(token="YOUR_HF_READ_TOKEN_HERE")
```
**Write Token (for uploading checkpoints)**
```
from huggingface_hub import login
login(token="YOUR_HF_WRITE_TOKEN_HERE")
```
**Note:** Replace with your actual Hugging Face tokens. You can generate these in your HF account settings under "Access Tokens".
**Important:** Ensure your HuggingFace account has been granted access to the Llama model before running the training script. Visit the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and request access if needed.

## Training Configurations

### Single-GPU DeepSpeed Training
```
!deepspeed --num_gpus=1 deepspeed_offload_finetune.py \
  --num_epochs 1 \
  --start_idx 0 \
  --end_idx 4000 \
  --batch_size 1 \
  --accum_steps 4 \
  --initial_epoch 0 \
  --hf_repo ash001/deepspeed-offload-llama-3.2-1B
```

### Resume from Checkpoint (if needed)
```
!deepspeed --num_gpus=1 deepspeed_offload_finetune.py \
  --num_epochs 1 \
  --start_idx 4000 \
  --end_idx 8000 \
  --batch_size 1 \
  --accum_steps 4 \
  --initial_epoch 0 \
  --hf_repo ash001/deepspeed-offload-llama-3.2-1B \
  --resume_file llama3.2-1B_0-4000-epoch-1.pt
```

**Note:** All training metrics will be automatically logged to your [W&B project dashboard](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune) for real-time monitoring.

### Parameter Explanation

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--num_epochs` | Number of training epochs | `1` |
| `--start_idx` | Data slice start index | `0` |
| `--end_idx` | Data slice end index | `4000` |
| `--batch_size` | Micro-batch size per GPU | `1` |
| `--accum_steps` | Gradient accumulation steps | `4` |
| `--initial_epoch` | Starting epoch for resume | `0` |
| `--hf_repo` | Hugging Face repository for uploads | Your repo name |
| `--resume_file` | Checkpoint file to resume from | Optional |

### Expected Outputs
- **Checkpoints:** Saved locally and uploaded to HF Hub automatically
- **W&B Tracking:** Real-time experiment tracking in [deepspeed-llama-3.2-1B-finetune](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune) project
- **Console Output:** Detailed training progress with step-by-step loss reporting
- **Model Artifacts:** Final model weights available on Hugging Face Hub

### Performance Notes
- **Memory Usage:** DeepSpeed ZeRO-2 with CPU offloading enables 1B+ parameter training on single 16GB GPU
- **Effective Batch Size:** 4 (1 per-GPU batch × 4 accumulation steps × 1 GPU)
- **CPU Offloading:** Optimizer states stored in CPU memory to maximize GPU memory for model parameters
- **Training Speed:** Mixed precision (FP16) and gradient accumulation optimize training throughput

### Model Evaluation
Use the provided evaluation notebook to assess model performance: [deepspeed-offload-model-evaluation.ipynb](https://www.kaggle.com/code/saratkannan/deepspeed-offload-model-evaluation)

**Key Evaluation Results:**
- **Loss Preservation:** Test data evaluation loss matches original pre-trained model performance
- **No Overfitting:** Model maintains generalization capabilities after fine-tuning
- **Memory Efficiency:** Successful training completion within hardware constraints

## Links & References

### Notebooks
- **Training Notebook:** [deepspeed-offload-llama-notebook.ipynb](https://www.kaggle.com/code/saratkannan/deepspeed-offload-llama-notebook) - Complete training configuration and setup
- **Evaluation Notebook:** [deepspeed-offload-model-evaluation.ipynb](https://www.kaggle.com/code/saratkannan/deepspeed-offload-model-evaluation) - Comprehensive loss comparison and performance assessment

### Model Artifacts
- **Fine-tuned Weights:** [ash001/deepspeed-offload-llama-3.2-1B](https://huggingface.co/ash001/deepspeed-offload-llama-3.2-1B/tree/main) - Final trained model checkpoint
- **Base Model:** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) - Original instruction-tuned LLaMA model

### Experiment Tracking
- **W&B Project:** [deepspeed-llama-3.2-1B-finetune](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune) - Complete training metrics and performance tracking
- **Run ID:** [clear-silence-11](https://wandb.ai/kannansarat9/deepspeed-llama-3.2-1B-finetune/runs/8jways6j) - Specific experiment run with detailed logs and training monitoring
