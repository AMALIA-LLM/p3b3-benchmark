# P3B3 - Portuguese Language Variant Bias Evaluation

A comprehensive framework for evaluating language model biases toward Portuguese language variants (European Portuguese vs Brazilian Portuguese) through multi-turn conversation generation and automated assessment.

## Overview

This project investigates whether large language models exhibit preferences or biases toward specific Portuguese language variants when generating responses. The system generates multi-turn conversations, evaluates them using multiple methods (classifier-based and LLM-as-judge), and analyzes performance across different models and prompt conditions.


## Project Structure

```
P3B3/
├── config/                  # Configuration files
│   └── settings.py         # Model and API settings
├── resources/              # Static resources
│   ├── all_prompts.json   # Multi-turn conversation prompts
│   └── image_markers/     # Provider logos for visualizations
├── src/                    # Source code
│   ├── analysis/          # Result aggregation and turn-level analysis
│   ├── annotation/        # Human annotation tools and agreement metrics
│   ├── dataset_analysis/  # Diversity metrics and conversation statistics
│   ├── evaluation/        # Generation and scoring pipelines
│   ├── models/            # Model backend implementations (API/VLLM/Ollama)
│   └── utils/             # Shared utilities
├── results/               # Generated model responses and scores
├── outputs/               # Analysis outputs and visualizations
├── run_scripts/         # SLURM job submission scripts
└── environment.yml        # Conda environment specification

```

## Installation

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate p3b3
```

### Configuration

Copy the template environment file and add your API keys:

```bash
# Copy the template
cp .env_copy .env

# Edit .env and replace with your actual API keys:
# - GOOGLE_API_KEY: Used by Gemini models
# - SABIA_API_KEY: Used by Sabia models
# - OLLAMA_BASE_URL: User for local Ollama models (default: http://localhost:11434)
```

Edit `config/settings.py` to adjust:
- `MAX_RETRIES`: API retry attempts
- `MAX_OUTPUT_TOKENS`: Generation token limit
- `MAX_MODEL_LEN`: VLLM context window
- `MAX_CONNECTIONS`: Concurrent API requests

## Usage

### 1. Generate Conversations

Generate multi-turn conversations with a language model:

```bash
# Using API model (e.g., Gemini)
python -m src.evaluation.generate --model-name-or-path google-langchain-api/gemini-3-flash-preview

# Using VLLM (requires GPU)
python -m src.evaluation.generate --model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct

# Using Ollama (local)
python -m src.evaluation.generate --model-name-or-path ollama/llama3
```

Generates 3 files per model (neutral, pt-pt, pt-br variants) in `results/<model_name>/`

### 2. Score Responses

#### Classifier-based Scoring

```bash
python -m src.evaluation.score_with_classifier results/<model_folder>
```

Outputs: `results/<model_folder>/class_scores/*.csv`

#### LLM Judge Scoring

```bash
python -m src.evaluation.score_with_llm \
    results/<model_folder> \
    --judge_name gemini-3-flash-preview \
    --max_connections 50 \
    --no-accumulate-context
```

Outputs: `results/<model_folder>/llm_scores/*.csv`

### 3. Aggregate Results

```bash
python -m src.analysis.aggregation
```

Creates:
- `results/combined_comprehensive_scores_llm_scores.csv` - All model scores
- `results/z_classifier_scores/` - Aggregated classifier results
- `results/z_llm_scores/` - Aggregated LLM judge results

### 4. Visualize Turn-Level Performance

```bash
python -m src.analysis.turn_analysis \
    results/z_classifier_scores/turn_level/aggregated_scores_by_turn_normal_all_prompts_PtVId.csv \
    --max-turns 3
```

Generates: `outputs/turn_progression_*.pdf`

## Supported Model Backends

### API Models
- **Gemini** (via LangChain): `google-langchain-api/gemini-3-flash-preview`
- **Sabia**: Set API key in `.env`

### VLLM Models
- Any Hugging Face model with CUDA support
- Examples: `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`

### Ollama Models
- Running local Ollama server required
- Format: `ollama/<model_name>`

## Evaluation Metrics

### Classifier Scores
Transformer-based models classify responses as European Portuguese (pt_pt) or Brazilian Portuguese (pt_br) with confidence scores (0-1).

### LLM Judge Scores
Gemini evaluates responses on a 0-10 scale with explanations for:
- Portuguese variant preference
- Response quality
- Linguistic markers


## Run and Slurm Scripts for Generating and Scoring

Simple script to run a complete pipeline using a VLLM model
```bash
# Generate conversations with VLLM
bash run_scripts/run_pipeline.sh <model_name>
```

`<model_name>` can be any Hugging Face model compatible with VLLM, e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, Gemini model, or ollama server.

SLURM scripts for batch processing multiple models:

```bash
# Submit single model job
sbatch run_scripts/run_single_model.sh <model_path>

# Submit all models
bash run_scripts/submit_all_models.sh

# Run LLM scoring (as a separate step to avoid rate limits)
sbatch run_scripts/run_llm_scoring.sh
```

Adapt the scripts to your cluster environment and model list.


## Citation

If you find this work relevant please cite:

```bibtex
@misc{p3b3_dataset,
  title={{P3B3}: A Multi-Turn Conversational Benchmark for Measuring {Portuguese} Variety Bias in {LLMs}},
  author={Ferreira, Rafael and Vieira, In{\^e}s and Calvo, In{\^e}s and Furtado, James and Paulo, Iago and Gl{\'o}ria-Silva, Diogo and Tavares, Diogo and Semedo, David and Magalh{\~a}es, Jo{\~a}o},
  year={2026},
}
```
