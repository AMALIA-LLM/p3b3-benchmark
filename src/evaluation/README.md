# Evaluation Module

This module provides tools for generating and evaluating multi-turn conversations with language models, with a focus on Portuguese language variant bias assessment (European Portuguese vs Brazilian Portuguese).

## Overview

The evaluation module consists of three main components:

1. **Response Generation** - Generate multi-turn conversations using various language models
2. **Classifier-based Scoring** - Evaluate responses using transformer-based classification models
3. **LLM-based Scoring** - Evaluate responses using another LLM as a judge


## Files

### `generate.py`
Handles conversation generation with language models.


**Usage:**
```bash
python -m src.evaluation.generate --model-name-or-path <model_path>

# Examples:
# Generate with Gemini API model
python -m src.evaluation.generate --model-name-or-path google-langchain-api/gemini-3-flash-preview

# Generate with vllm (needs GPU)
python -m src.evaluation.generate --model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct
```

### `score_with_classifier.py`
Evaluates responses using transformer-based text classification models.


**Usage:**
```bash
python -m src.evaluation.score_with_classifier <base_dir>

# Example:
python -m src.evaluation.score_with_classifier results/google-langchain-api-gemini-3-flash-preview
```

### `score_with_llm.py`
Evaluates generated responses using an LLM judge (typically Gemini).

**Usage:**
```bash
python -m src.evaluation.score_with_llm \
    <input_folder> \
    --judge_name gemini-3-flash-preview \
    --max_connections 50 \
    [--no-accumulate-context]

# Example:
python -m src.evaluation.score_with_llm \
    results/google-langchain-api-gemini-3-flash-preview \
    --judge_name gemini-3-flash-preview \
    --max_connections 50 \
    --no-accumulate-context
```


### `prompts.py`
Contains evaluation prompts, primarily for Portuguese language variant assessment.


## Workflow

### Standard Evaluation Pipeline

1. **Generate Conversations**
2. **Score with Classifier**
3. **Score with LLM Judge**


## Supported Models

The evaluation module supports various model backends:
- **API Models**: Gemini (using LangChain), Sabia (set respective API keys in env file)
- **Local Models**: Ollama models already running locally (set respective API keys in env file)
- **VLLM Models**: High-performance inference with models Hugging Face

## Output Format

### Classifier Scoring Output
List of dictionaries mapping label names to scores:
```python
[
    {"pt_br": 0.8, "pt_pt": 0.2},
    {"pt_br": 0.3, "pt_pt": 0.7}
]
```

### LLM Scoring Output
CSV with columns:
- `model_name` - Name of the evaluated model
- `conversation_id` - Unique conversation identifier
- `turn_idx` - Turn number in conversation
- `context` - Cumulative conversation context
- `response` - Model's response
- `prompt_type` - Type of prompt used
- `score` - Numerical score (0-10 for PT variant)
- `explanation` - Judge's explanation
- `timestamp` - Evaluation timestamp
