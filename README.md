# P3B3: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in LLMs

[![Code](https://img.shields.io/badge/GitHub-P3B3%20Benchmark-blue?logo=github)](https://github.com/AMALIA-LLM/p3b3-benchmark)
[![Paper: ACL Anthology](https://img.shields.io/badge/Paper-Link-red)](https://arxiv.org/abs/2606.16753)
[![MeLLM @ ACL 2026](https://img.shields.io/badge/Workshop-MeLLM%20%40%20ACL%202026-green)](https://www.mellm.org/)

This repository provides the code and resources for running the P3B3 benchmark, introduced in the paper
[P3B3: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in LLMs](https://arxiv.org/abs/2606.16753v1), accepted at the [MeLLM Workshop](https://www.mellm.org/) at ACL 2026.

P3B3 is a benchmark for evaluating language model biases toward Portuguese language variants (European Portuguese vs Brazilian Portuguese) through multi-turn conversation generation and automated assessment.

## Overview

This work investigates whether large language models exhibit preferences or biases toward specific Portuguese language variants when generating responses. The system generates multi-turn conversations, evaluates them using multiple methods (classifier-based and LLM-as-judge), and analyzes performance across different models and prompt conditions.


## Code Structure

```
P3B3/
├── config/                  # Configuration files
│   └── settings.py          # Model and API settings
├── resources/               # Static resources
│   ├── all_prompts.json     # Multi-turn conversation prompts
│   └── image_markers/       # Provider logos for visualizations
├── src/                     # Source code
│   ├── analysis/            # Result aggregation and turn-level analysis
│   ├── annotation/          # Human annotation tools and agreement metrics
│   ├── dataset_analysis/    # Diversity metrics and conversation statistics
│   ├── evaluation/          # Generation and scoring pipelines
│   ├── models/              # Model backend implementations (API/VLLM/Ollama)
│   └── utils/               # Shared utilities
├── results/                 # Generated model responses and scores
├── outputs/                 # Analysis outputs and visualizations
├── run_scripts/             # SLURM job submission scripts
└── environment.yml          # Conda environment specification

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

# Edit .env and replace with your actual API keys as needed:
# - GEMINI_API_KEY: Used by Gemini models
# - MARITACA_API_KEY: Used by Sabia models
# - OLLAMA_BASE_URL: User for local Ollama models
```

Edit `config/settings.py` to adjust:
- `MAX_RETRIES`: API retry attempts
- `MAX_OUTPUT_TOKENS`: Generation token limit
- `MAX_MODEL_LEN`: VLLM context window
- `MAX_CONNECTIONS`: Concurrent API requests

## Usage

### 1. Generate Conversations

Generate multi-turn responses using a language model:

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
- **Gemini** (via LangChain): e.g. `google-langchain-api/gemini-3-flash-preview`
- **Sabia**: Set API key in `.env`

### VLLM Models
- Any Hugging Face model with CUDA support
- Examples: `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`

### Ollama Models
- Running local Ollama server required
- Format: `ollama/<model_name>`

## Evaluation Metrics

### Classifier Scores
Transformer-based models classify responses as European Portuguese (pt_pt) or Brazilian Portuguese (pt_br) with probability scores (0-1).

### LLM Judge Scores
Gemini evaluates responses on a 0-10 scale with explanations for:
- Portuguese variant preference
- Linguistic markers


## Run and Slurm Scripts for Generating and Scoring

Simple script to run a complete pipeline using vLLM
```bash
# Generate conversations with vLLM
bash run_scripts/run_pipeline.sh <model_name>
```

`<model_name>` can be any Hugging Face model compatible with vLLM, e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, Gemini model, or ollama server.

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
@inproceedings{ferreira_p3b3,
      title={{P3B3}: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in {LLMs}}, 
      author={Rafael Ferreira and Inês Vieira and Inês Calvo and James Furtado and Iago Paulo and Diogo Tavares and Diogo Glória-Silva and David Semedo and João Magalhães},
      booktitle={Proceedings of the 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM)},
      year={2026},
      publisher={Association for Computational Linguistics},
      eprint={2606.16753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.16753}, 
}
```
