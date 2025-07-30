# DeepSpeed Pipeline Parallelism

## Objective
**Objective:** Fine-tune a [1B-parameter LLaMA model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) using DeepSpeed's Pipeline Parallelism to enable efficient training across multiple GPUs by distributing model layers across devices and overlapping computation with communication.

## Technique & Tools
- **Technique:** DeepSpeed Pipeline Parallelism with 2-stage pipeline distribution, ZeRO-1 optimization for optimizer state sharding, and mixed precision training (FP16). Used uniform layer partitioning to split the model into two equal stages across GPUs.
- **Tools:** DeepSpeed Pipeline API with PipelineModule, Weights & Biases for experiment tracking (project: [llama-1b-ds-pipeline](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline)), Hugging Face Transformers and Hub for model management, and NCCL backend for inter-GPU communication.

## Implementation Details
### Pipeline Configuration:
* **Pipeline Stages:** 2-stage uniform partitioning with embedding + first half of decoder layers on stage 1, remaining layers + final norm + LM head on stage 2
* **ZeRO Optimization:** ZeRO-1 stage for optimizer state sharding across GPUs
* **Mixed Precision:** FP16 training with automatic loss scaling and gradient clipping
* **Custom Pipeline Modules:** Implemented specialized pipeline wrappers (EmbeddingPipe, DecoderLayerPipe, FinalNormPipe, LMHeadPipe) for proper tensor flow between stages
* **Hardware:** 2 × T4 16GB GPUs on Kaggle Notebooks
* **Dataset:** Fine-tuned on [arxiv abstracts](https://huggingface.co/datasets/ash001/arxiv-abstract) with configurable data range slicing
* **Gradient Handling:** Gradient clipping (max_norm=1.0) and accumulation with pipeline-aware scheduling

### Custom Pipeline Architecture:
* **EmbeddingPipe:** Handles token embeddings and input preprocessing
* **DecoderLayerPipe:** Wraps individual transformer layers with rotary position embeddings
* **FinalNormPipe:** Applies layer normalization before language modeling head
* **LMHeadPipe:** Computes logits and handles loss calculation with shifted cross-entropy

## Results & Evaluation
### Training Performance:
* **Model Quality:** Achieved similar evaluation loss on test data as the original pre-trained model, indicating successful knowledge transfer without overfitting
* **Pipeline Efficiency:** Successfully distributed 1B parameter model across 2 GPUs with overlapped computation between pipeline stages
* **Inference:** Model inference capabilities remain unchanged from base model - no modifications to inference pipeline
* **Memory Distribution:** Pipeline parallelism enabled training by splitting model layers rather than parameters, complementing ZeRO-1 optimizer sharding

### Training Metrics:
* Pipeline scheduling maintained consistent throughput across stages
* Mixed precision training with loss scaling prevented gradient underflow
* No pipeline bubble issues observed during training

## MLOps Practices
### Experiment Tracking
* **Weights & Biases Integration:** Comprehensive experiment logging with project [llama-1b-ds-pipeline](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline) and specific run [zany-firefly-41](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline/runs/0drdrxdg). Tracked training loss, epoch metrics, perplexity, samples processed, and total training time with step-by-step monitoring.
* **Real-time Monitoring:** Live tracking of loss curves, pipeline stage utilization, and gradient statistics through W&B dashboard for training insights and performance optimization.

### Model Versioning
* **Native Checkpoint Format:** DeepSpeed pipeline checkpoint shards uploaded directly to Hugging Face Hub at [ash001/deepspeed-pipeline-llama-1B](https://huggingface.co/ash001/deepspeed-pipeline-llama-1B/tree/main) - requires conversion for standard inference
* **Checkpoint Structure:** Contains 19 layer shards (`layer_00-model_states.pt` to `layer_18-model_states.pt`), 2 pipeline stage shards (`mp_rank_00_model_states.pt`, `mp_rank_01_model_states.pt`), and ZeRO optimizer states (`zero_pp_rank_0_mp_rank_00_optim_states.pt`, `zero_pp_rank_0_mp_rank_01_optim_states.pt`) 
* **Tokenizer Integration:** Complete tokenizer configuration uploaded alongside checkpoint shards for model compatibility

## How to Run
### Prerequisites
- **Platform:** Kaggle Notebooks with GPU acceleration (2 × T4 recommended)
- **Python 3.8+** with PyTorch and DeepSpeed support
- **Hugging Face Account** with write access token for model uploads
- **Weights & Biases Account** for experiment tracking

### Setup Instructions
#### 1. Clone Repository and Navigate
```
!git clone https://github.com/sparklerz/multigpu-llm-finetuning.git
%cd multigpu-llm-finetuning
%cd deepspeed-pipeline
```
#### 2. Install Dependencies
```
!pip install -r requirements.txt
```
#### 3. Authenticate with Hugging Face
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

#### 4. Set Environment Variable for HF Token
```
%env HF_TOKEN=your_hf_write_token_here
```
**Note:** Replace `your_hf_write_token_here` with your actual HuggingFace write token.

#### 5. Verify Token Setup (Optional)
```
import os, subprocess, textwrap
print("HF_TOKEN =", os.getenv("HF_TOKEN"))
```
#### 6. Authenticate with Weights & Biases
```
!wandb login your_wandb_api_key_here
```
**Note:** Replace `your_wandb_api_key_here` with your actual W&B API key from your account settings.

## Training Configuration

### DeepSpeed Pipeline Training
```
!deepspeed --num_gpus=2 deepspeed_pipeline_finetune_llama_new.py \
  --num_epochs 3 \
  --start_idx 0 \
  --end_idx 25000 \
  --batch_size 2 \
  --accum_steps 16 \
  --hf_repo ash001/deepspeed-pipeline-llama-1B
```

**Note:** All training metrics will be automatically logged to your [W&B project dashboard](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline) for real-time monitoring.

### Parameter Explanation

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--num_epochs` | Number of training epochs | `3` |
| `--start_idx` | Data slice start index | `0` |
| `--end_idx` | Data slice end index | `25000` |
| `--batch_size` | Micro-batch size per GPU | `2` |
| `--accum_steps` | Gradient accumulation steps | `16` |
| `--hf_repo` | Hugging Face repository for uploads | Your repo name |

### Expected Outputs
- **Pipeline Checkpoints:** DeepSpeed format checkpoints with stage-specific weights
- **W&B Tracking:** Real-time experiment monitoring in llama-1b-ds-pipeline project
- **HuggingFace Upload:** Automatic model and tokenizer upload to specified repository
- **Training Logs:** Per-rank logging with pipeline stage performance metrics

### Performance Characteristics
- **Pipeline Efficiency:** 2-stage model splitting with overlapped forward/backward passes
- **Memory Usage:** ZeRO-1 optimizer state sharding across GPUs with pipeline model partitioning
- **Communication:** Efficient activation passing between pipeline stages with minimal overhead
- **Throughput:** Pipeline parallelism reduces idle time through computation overlap

### Model Evaluation
Use the provided evaluation notebook for model assessment:
- **Evaluation Notebook:** [deepspeed-pipeline-evaluation-notebook.ipynb](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-evaluation-notebook) - Comprehensive loss comparison and text generation quality assessment

## Using the Uploaded Checkpoints

**Important:** The uploaded model files are in DeepSpeed pipeline checkpoint format and cannot be directly loaded with standard HuggingFace methods.

### For Inference Use:
- Use the conversion script provided in the [evaluation notebook](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-evaluation-notebook) to merge checkpoint shards
- The conversion process combines all layer shards and pipeline stages into a unified PyTorch state dict
- Load the converted checkpoint into a standard LLaMA model for inference

### Checkpoint Structure:
- **Layer Shards:** `layer_00-model_states.pt` through `layer_18-model_states.pt` (19 files) - individual transformer layers
- **Pipeline Stages:** `mp_rank_00_model_states.pt`, `mp_rank_01_model_states.pt` - pipeline stage distributions  
- **Optimizer States:** `zero_pp_rank_0_mp_rank_00_optim_states.pt`, `zero_pp_rank_0_mp_rank_01_optim_states.pt` - ZeRO-1 optimizer state shards
- **Tokenizer Files:** Standard HuggingFace tokenizer configuration

**Key Findings:**
- **Loss Consistency:** Test data evaluation loss matches original pre-trained model performance
- **No Degradation:** Model maintains original capabilities after pipeline parallel training
- **Checkpoint Format:** Uploaded weights are in DeepSpeed pipeline format and require conversion for standard inference (conversion script provided in [evaluation notebook](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-evaluation-notebook))

## Links & References

### Notebooks
- **Training Notebook:** [deepspeed-pipeline-notebook.ipynb](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-notebook) - Complete pipeline parallelism setup and training workflow
- **Evaluation Notebook:** [deepspeed-pipeline-evaluation-notebook.ipynb](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-evaluation-notebook) - Model loss comparison and performance assessment

### Model Artifacts
- **Pipeline Checkpoint Shards:** [ash001/deepspeed-pipeline-llama-1B](https://huggingface.co/ash001/deepspeed-pipeline-llama-1B/tree/main) - DeepSpeed native format requiring conversion for inference (see [evaluation notebook](https://www.kaggle.com/code/saratkannan/deepspeed-pipeline-evaluation-notebook) for conversion script)  
- **Base Model:** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) - Original LLaMA-3.2-1B instruction-tuned model

### Experiment Tracking
- **W&B Project:** [llama-1b-ds-pipeline](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline) - Complete training metrics, loss curves, and system performance
- **Run ID:** [zany-firefly-41](https://wandb.ai/kannansarat9/llama-1b-ds-pipeline/runs/0drdrxdg) - Specific pipeline training run with detailed pipeline stage monitoring
