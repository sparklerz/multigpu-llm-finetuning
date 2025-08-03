# LLM Foundry FSDP Fine-tuning

## Objective
**Objective:** Fine-tune a [1.3B-parameter OPT model](https://huggingface.co/facebook/opt-1.3b) using MosaicML's LLM Foundry framework with PyTorch FSDP to enable efficient distributed training across multiple GPUs while leveraging pre-built, production-ready fine-tuning configurations.

## Technique & Tools
- **Technique & Tools:** Used MosaicML's LLM Foundry with PyTorch Fully Sharded Data Parallel (FSDP) for distributed training across 2 GPUs. LLM Foundry provides config-based fine-tuning with built-in FSDP sharding strategies, mixed precision training (amp_fp16), and activation checkpointing. Training code leverages Foundry's pre-built training loop with custom YAML configurations. Tracked experiment metrics with Weights & Biases (project: [llm-foundry-demo](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace), run: [opt_dolly_sft](https://wandb.ai/kannansarat9/llm-foundry-demo/runs/t1i96kr7)).

## Implementation Details
### LLM Foundry Configuration:
* **Framework:** MosaicML LLM Foundry v0.6.0 with config-based training setup
* **FSDP Strategy:** `FULL_SHARD` to distribute model parameters, gradients, and optimizer states across GPUs
* **Mixed Precision:** `amp_fp16` for memory efficiency and faster training
* **Memory Optimization:** Activation checkpointing enabled, all-gather optimization with `limit_all_gathers: true`
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

## Model Evaluation

### Key Evaluation Results:
* **Token Accuracy:** Progressive improvement from baseline to epoch completion, with evaluation accuracy showing consistent upward trend
* **Language Perplexity:** Demonstrated reduction in perplexity scores during training, indicating better language modeling capability
* **Cross-Entropy Loss:** Validation loss showed steady decline across training steps, confirming model learning without overfitting
* **Evaluation Frequency:** Model evaluated every 100 batches (`eval_interval: 100ba`) for continuous performance monitoring

### Evaluation Setup:
* **Validation Split:** 10% of Dolly-15K dataset (approximately 1,500 examples) used for evaluation
* **Metrics Tracked:** Token-level accuracy, language perplexity, and cross-entropy loss for both training and validation sets
* **Evaluation Data:** Same prompt/response format as training with `train_on_prompt: false` to evaluate only response generation quality
* **Monitoring:** Real-time evaluation metrics logged to W&B dashboard for immediate performance insights

![Training and evaluation metrics showing consistent improvement across both epochs with stable convergence](https://github.com/user-attachments/assets/c42c12a3-abbc-4d28-9c7f-f0899c19eb83)

## MLOps Practices
### Experiment Tracking
* **Experiment Tracking:** Comprehensive logging with Weights & Biases integration in project [llm-foundry-demo](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace) with run ID [opt_dolly_sft](https://wandb.ai/kannansarat9/llm-foundry-demo/runs/t1i96kr7). LLM Foundry automatically tracks training/validation loss, learning rate schedules, speed metrics, and model parameters. Built-in W&B callbacks provide real-time monitoring of training progress and evaluation results.

### Model Versioning
**Multi-tier checkpoint strategy implemented:**
* **Local Checkpoints:** HuggingFace format checkpoints saved in `./checkpoints` with epoch-based folder structure (`ep1`, `ep2`)
* **Branch Versioning:** Final model weights saved to separate repository branches for each epoch:
  * [epoch-1 branch](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B/tree/epoch-1)
  * [epoch-2 branch](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B/tree/epoch-2)
* **Separate Repositories:** Individual HuggingFace repositories for each training epoch:
  * [llm-foundry-fsdp-opt-1.3B-epoch-1](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B-epoch-1/tree/main)
  * [llm-foundry-fsdp-opt-1.3B-epoch-2](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B-epoch-2/tree/main)

## How to Run
### Prerequisites
- **Platform:** Kaggle Notebooks with GPU acceleration (2 × T4 recommended)
- **Python 3.8+** with LLM Foundry dependencies
- **Hugging Face Account** with write access token
- **Weights & Biases Account** for experiment tracking

### Setup Instructions
#### 1. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.git
%cd multigpu-llm-finetuning
%cd llm-foundry-finetune
```
#### 2. Install Dependencies
```
!pip install -r requirements.txt
```
#### 3. Clone and Install LLM Foundry
```
!git clone https://github.com/mosaicml/llm-foundry.git llm-foundry
!pip install -e llm-foundry
```
#### 4. Authenticate with Hugging Face
```
from huggingface_hub import login
login(token="your_hf_token_here") # Replace with your actual write token
```
#### 5. Authenticate with Weights & Biases
```
!wandb login your_wandb_api_key_here  # Replace with your actual API key
```
**Note:** Replace with your actual W&B API key. You can find this in your W&B account settings.

#### 6. Prepare Dataset
```
!python prepare_dolly.py
```
#### 7. Run Fine-tuning
```
!composer -n 2 \
  llm-foundry/scripts/train/train.py \
  configs/finetune_opt.yaml
```
**Command Parameters:**
- `composer` - MosaicML's distributed training launcher
- `-n 2` - Number of GPUs to use (2 × T4)
- `llm-foundry/scripts/train/train.py` - LLM Foundry's training script
- `configs/finetune_opt.yaml` - Configuration file containing all training parameters

#### 8. Upload Checkpoints to Hugging Face Hub
**Option A: Separate Repositories (Simpler)**
```
# For Epoch 1
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./checkpoints/huggingface/ep1", 
                                           torch_dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/huggingface/ep1", 
                                        local_files_only=True)

model.push_to_hub("ash001/llm-foundry-fsdp-opt-1.3b-epoch-1")
tokenizer.push_to_hub("ash001/llm-foundry-fsdp-opt-1.3b-epoch-1")

# For Epoch 2  
model = AutoModelForCausalLM.from_pretrained("./checkpoints/huggingface/ep2", 
                                           torch_dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/huggingface/ep2", 
                                        local_files_only=True)

model.push_to_hub("ash001/llm-foundry-fsdp-opt-1.3B-epoch-2")
tokenizer.push_to_hub("ash001/llm-foundry-fsdp-opt-1.3B-epoch-2")
```
**Option B: Branch-based Repository (More Organized)**
```
import tempfile
from pathlib import Path
from huggingface_hub import Repository
from transformers import AutoModelForCausalLM, AutoTokenizer

# Customize these variables per epoch
EPOCH = 1  # Change to 2 for second epoch
CHECKPOINT_DIR = f"./checkpoints/huggingface/ep{EPOCH}"
REPO_ID = "ash001/llm-foundry-fsdp-opt-1.3b"
BRANCH_NAME = f"epoch-{EPOCH}"

# Load checkpoint
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, torch_dtype="auto", 
                                           local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, local_files_only=True)

# Clone repo and create branch
tmp_dir = Path(tempfile.mkdtemp())
repo = Repository(local_dir=tmp_dir, clone_from=REPO_ID)
repo.git_checkout(BRANCH_NAME, create_branch_ok=True)

# Save and push
model.save_pretrained(tmp_dir, safe_serialization=True)
tokenizer.save_pretrained(tmp_dir)
repo.push_to_hub(commit_message=f"Add weights after epoch {EPOCH}")

print(f"✅ Epoch {EPOCH} pushed to {REPO_ID} (branch: {BRANCH_NAME})")
```

### Configuration Details
The `configs/finetune_opt.yaml` file contains all training parameters:

| Configuration | Value | Description |
|-----------|-------------|----------------|
| Model | facebook/opt-1.3b | Base model for fine-tuning |
| Max Duration | 2ep | Training for 2 complete epochs |
| Global Batch Size | 16 | Effective batch size across all GPUs |
| Learning Rate | 1e-6 | Conservative LR for stable fine-tuning |
| FSDP Strategy | FULL_SHARD | Complete parameter/gradient sharding |
| Mixed Precision | amp_fp16 | FP16 for memory and speed optimization |
| Activation Checkpointing | true | Memory saving through recomputation |

### Expected Outputs
- **Checkpoints:** Local saves in `./checkpoints/` with epoch-based organization
- **W&B Tracking:** Real-time metrics in [llm-foundry-demo](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace) project
- **HF Integration:** Automatic upload to configured HuggingFace repositories
- **Evaluation Logs:** Built-in Foundry evaluator results logged every 100 batches

## Links & References

### Notebooks
- **Training Notebook:** [llm-foundry-notebook.ipynb](https://www.kaggle.com/code/saratkannan/llm-foundry-notebook) - Complete LLM Foundry fine-tuning workflow with FSDP setup

### Model Artifacts
- **Base Model:** [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b) - Original OPT-1.3B model
- **Fine-tuned Weights (Branches):**
 - [Epoch 1](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B/tree/epoch-1)
 - [Epoch 2](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B/tree/epoch-2)
- **Fine-tuned Weights (Separate Repos):**
 - [llm-foundry-fsdp-opt-1.3B-epoch-1](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B-epoch-1/tree/main)
 - [llm-foundry-fsdp-opt-1.3B-epoch-2](https://huggingface.co/ash001/llm-foundry-fsdp-opt-1.3B-epoch-2/tree/main)

### Experiment Tracking
- **W&B Project:** [llm-foundry-demo](https://wandb.ai/kannansarat9/llm-foundry-demo/workspace) - Complete training metrics and LLM Foundry evaluation results
- **Run ID:** [opt_dolly_sft](https://wandb.ai/kannansarat9/llm-foundry-demo/runs/t1i96kr7) - Specific experiment run with detailed loss curves and performance metrics
