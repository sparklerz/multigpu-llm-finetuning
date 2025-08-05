# Ray Tune Hyperparameter Optimization

## Objective
**Objective:** Use Ray Tune to run 6 different trials to find the optimal hyperparameter combination (involving learning rate, batch size, weight decay, epochs, warmup steps) that results in the lowest evaluation loss for fine-tuning a [0.5B-parameter Qwen2 series language model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).

## Technique & Tools
- **Technique:** Ray Tune hyperparameter optimization with ASHA (Asynchronous Successive Halving Algorithm) scheduler for early stopping of poorly performing trials. Used distributed training across multiple GPUs with automatic resource allocation and concurrent trial execution.
- **Tools:**
    - Ray Tune for hyperparameter search space exploration and trial management
    - ASHA Scheduler for efficient early stopping based on evaluation loss
    - PyTorch Transformers for model training and tokenization
    - Weights & Biases for experiment tracking (project: [ray-tune-qwen](https://wandb.ai/kannansarat9/ray-tune-qwen/workspace))
    - Hugging Face Hub for model versioning and deployment
    - [IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (20% subset) for fine-tuning with text truncation to 256 tokens

## Implementation Details
### Hyperparameter Search Space:
* **Learning Rate:** Log-uniform distribution between `5e-6 and 5e-5`
* **Batch Size:** Choice between `[1, 2, 4]` with gradient accumulation to maintain effective batch size of 32
* **Weight Decay:** Uniform distribution between `0.0 and 0.3`
* **Epochs:** Choice between `[1, 2, 3]` training epochs
* **Warmup Steps:** Choice between `[0, 100, 200]` warmup steps

### Training Configuration:
* **Hardware:** 2 × T4 16GB GPUs enabling 2 concurrent trials
* **Resource Allocation:** 2 CPUs and 1 GPU per trial with automatic Ray resource management
* **Memory Optimization:** Gradient checkpointing enabled, mixed precision training (BF16/FP16)
* **Dataset Split:** IMDb train data with 10% held out for validation
* **Scheduler:** ASHA with grace period of 1 epoch, reduction factor of 2, and maximum of 3 epochs

### Trial Management:
* **Total Trials:** 6 hyperparameter combinations explored
* **Early Stopping:** ASHA scheduler terminates underperforming trials based on evaluation loss
* **Checkpointing:** Ray Air checkpoints saved per epoch for trial resumption
* **Reproducibility:** Fixed seed (42) across all trials for consistent results

## Results & Evaluation
### Hyperparameter Optimization Results:
* **Evaluation Method:** The best hyperparameter combination was chosen by selecting the model with the lowest evaluation loss across all completed trials.
* **Trial Performance:** Out of 6 trials initiated, ASHA scheduler efficiently terminated poorly performing configurations early, allowing computational resources to focus on promising hyperparameter combinations.
* **Best Configuration:** The optimal hyperparameters were automatically identified and used to generate the final fine-tuned model uploaded to Hugging Face Hub.

### Training Efficiency:
* **Resource Utilization:** Concurrent trial execution on 2 GPUs maximized hardware utilization
* **Time Savings:** ASHA early stopping prevented wasting compute on suboptimal configurations
* **Memory Management:** Gradient checkpointing and mixed precision enabled stable training within 16GB GPU memory limits

## MLOps Practices
### Experiment Tracking
* **Weights & Biases Integration:** Comprehensive experiment logging with project [ray-tune-qwen](https://wandb.ai/kannansarat9/ray-tune-qwen/workspace). Each trial was tracked with individual run IDs:
    * Trial Runs: [tune-47ee3_00000](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/2zodp70e), [tune-47ee3_00001](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/yyp1seic), [tune-47ee3_00002](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/8rstimla), [tune-47ee3_00003](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/5iojbkih), [tune-47ee3_00004](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/mavylmsq), [tune-47ee3_00005](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/o88opgs3)
    * Metrics Logged: Training loss, evaluation loss, epoch progress, and hyperparameter configurations
    * Real-time Monitoring: Live tracking of trial performance for early identification of optimal configurations
* **Model Deployment:** Best performing trial's checkpoint automatically pushed to [ash001/ray-tune-qwen-0.5B](https://huggingface.co/ash001/ray-tune-qwen-0.5B/tree/main) with complete model weights and tokenizer configuration.

## How to Run
### Prerequisites
- **Platform:** Kaggle Notebooks with GPU acceleration (2 × T4 recommended) or equivalent multi-GPU environment
- **Python 3.8+** with **Ray 2.34.0** and PyTorch support
- **Hugging Face Account** with write access token for model uploads
- **Weights & Biases Account** for experiment tracking

### Setup Instructions
#### 1. Verify Python Version
```
!python --version
```
#### 2. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.git
%cd multigpu-llm-finetuning
%cd ray-tune
```
#### 3. Install Dependencies
```
!pip install -qU -r requirements.txt --no-cache-dir
```
#### 4. Authenticate with Hugging Face
```
from huggingface_hub import login
login(token="your_hf_write_token_here")  # Replace with your actual write token
```
#### 5. Set Environment Variable for HF Token
```
!export HF_TOKEN=your_hf_write_token_here
```
**Note:** Replace `your_hf_write_token_here` with your actual HuggingFace write token.
#### 6. Authenticate with Weights & Biases
```
!wandb login your_wandb_api_key_here  # Replace with your actual API key
```
**Note:** Replace with your actual W&B API key. You can find this in your W&B account settings.
#### 7. Execute Hyperparameter Search
```
!python ray_tune_qwen_llm.py
```

### Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------------|----------------|
| `num_samples` | 6 | Total number of hyperparameter trials |
| `max_t` | 3 | Maximum epochs per trial |
| `grace_period` | 1 | Minimum epochs before early stopping |
| `resources_per_trial` | {"cpu": 2, "gpu": 1} | Hardware allocation per trial |

### Expected Outputs
- **Console Output:** Real-time trial progress with hyperparameter configurations and evaluation losses
- **W&B Dashboard:** Live experiment tracking with trial comparison and metrics visualization
- **Model Artifacts:** Best performing model automatically uploaded to Hugging Face Hub
- **Best Configuration:** Printed summary of optimal hyperparameters and corresponding evaluation loss

### Performance Monitoring
- **Trial Status:** Ray Tune dashboard shows real-time trial execution and resource utilization
- **Early Stopping:** ASHA scheduler logs show which trials were terminated early and why
- **Resource Usage:** GPU and memory utilization tracked across concurrent trials

## Links & References

### Notebooks
- **Training Notebook:** [ray-tune-qwen-0.5B-notebook-6-trials.ipynb](https://www.kaggle.com/code/saratkannan/ray-tune-qwen-0-5b-notebook-6-trials) - Complete Ray Tune workflow with 6 trial hyperparameter search

### Model Artifacts
- **Fine-tuned Weights:** [ash001/ray-tune-qwen-0.5B](https://huggingface.co/ash001/ray-tune-qwen-0.5B/tree/main) - Best performing model from hyperparameter optimization
- **Base Model:** [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) - Original Qwen2-0.5B instruction-tuned model

### Experiment Tracking
- **W&B Project:** [ray-tune-qwen](https://wandb.ai/kannansarat9/ray-tune-qwen/workspace) - Complete hyperparameter search results and trial comparisons
- **Trial Runs:** Individual experiment tracking for all 6 trials
    - [tune-47ee3_00000](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/2zodp70e)
    - [tune-47ee3_00001](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/yyp1seic)
    - [tune-47ee3_00002](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/8rstimla)
    - [tune-47ee3_00003](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/5iojbkih)
    - [tune-47ee3_00004](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/mavylmsq)
    - [tune-47ee3_00005](https://wandb.ai/kannansarat9/ray-tune-qwen/runs/o88opgs3)
