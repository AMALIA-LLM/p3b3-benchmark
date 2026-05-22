#!/bin/bash

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=p3b3_gen
#SBATCH --output=./joblog/%x-%j.out
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-11:00:00


eval "$(conda shell.bash hook)"

# activate conda environment
conda activate p3b3

# go to folder
cd "$HOME/P3B3"

# Get model name and GPU count from command-line arguments
MODEL="$1"
GPU_COUNT="${2:-1}"  # Default to 1 GPU if not specified

# Check if MODEL is provided
if [ -z "$MODEL" ]; then
    echo "ERROR: Model name not provided"
    echo "Usage: sbatch run_single_model.sh <model_name> [gpu_count]"
    exit 1
fi

# Sanitize model name: replace / with - and remove special characters
MODEL_SANITIZED=$(echo "$MODEL" | sed 's/\//-/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

# Set output directory
OUT_DIR="results/${MODEL_SANITIZED}"

echo "=============================================================="
echo "Model: $MODEL"
echo "GPUs requested: $GPU_COUNT"
echo "Output directory: $OUT_DIR"
echo "=============================================================="

# Generate for the model
echo ""
echo "Starting generation..."
PYTHONUNBUFFERED=1 python -m src.evaluation.generate --model-name-or-path "$MODEL" --base-output-dir "$OUT_DIR" --max_connections 50

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed for model $MODEL"
    exit 1
fi

echo ""
echo "Generation completed successfully!"
echo ""

# Run scoring/evaluation
echo "Starting evaluation..."
PYTHONUNBUFFERED=1 python -m src.evaluation.score_with_classifier "$OUT_DIR"

# Run with LLM scoring
# TODO commented to avoid rate limits during generation (uncomment if needed or use run_llm_scoring.sh as a separate step)
# echo ""
# echo "Starting LLM scoring..."
# PYTHONUNBUFFERED=1 python -m src.evaluation.score_with_llm "$OUT_DIR" --max_connections 50 --no-accumulate-context

# Check if scoring was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Scoring failed for model $MODEL"
    exit 1
fi

echo ""
echo "=============================================================="
echo "Completed: $MODEL"
echo "Results saved in: $OUT_DIR"
echo "=============================================================="
