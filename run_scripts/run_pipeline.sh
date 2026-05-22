#!/bin/bash

# Full P3B3 Pipeline: Generation -> Classifier Scoring -> LLM Judge -> Aggregation + Turn-level Analysis
# Usage: bash run_pipeline.sh [model_id]
# Default model: meta-llama/Llama-3.1-8B-Instruct

set -e  # Exit on any error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate p3b3

# Get model name from command-line argument, default to Llama 3.1 8B Instruct
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"

echo "=============================================================="
echo "P3B3 Full Pipeline"
echo "=============================================================="
echo "Model: $MODEL"
echo "=============================================================="
echo ""

# Sanitize model name for directory: replace / with - and remove special characters
MODEL_SANITIZED=$(echo "$MODEL" | sed 's/\//-/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

# Set output directory
OUT_DIR="results/${MODEL_SANITIZED}"

echo "Step 1/5: Generation"
echo "=============================================================="
echo "Generating responses for model: $MODEL"
echo "Output directory: $OUT_DIR"
echo ""

# Run generation (creates results/${MODEL_SANITIZED}/*.json files)
PYTHONUNBUFFERED=1 python -m src.evaluation.generate \
    --model-name-or-path "$MODEL" \
    --base-output-dir "$OUT_DIR" \
    --max_connections 50

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed for model $MODEL"
    exit 1
fi

echo ""
echo "Generation completed successfully!"
echo ""

echo "Step 2/5: Classifier Scoring"
echo "=============================================================="
echo "Running classifier-based scoring (PtVId + PeroVaz)"
echo ""

# Run classifier scoring (creates results/${MODEL_SANITIZED}/class_scores/*.csv files)
PYTHONUNBUFFERED=1 python -m src.evaluation.score_with_classifier "$OUT_DIR"

# Check if classifier scoring was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Classifier scoring failed for model $MODEL"
    exit 1
fi

echo ""
echo "Classifier scoring completed successfully!"
echo ""

echo "Step 3/5: LLM Judge Scoring"
echo "=============================================================="
echo "Running LLM judge evaluation"
echo ""

# Run LLM judge scoring (creates results/${MODEL_SANITIZED}/llm_scores/*.csv files)
PYTHONUNBUFFERED=1 python -m src.evaluation.score_with_llm "$OUT_DIR" \
    --max_connections 50 \
    --no-accumulate-context

# Check if LLM scoring was successful
if [ $? -ne 0 ]; then
    echo "ERROR: LLM scoring failed for model $MODEL"
    exit 1
fi

echo ""
echo "LLM judge scoring completed successfully!"
echo ""

echo "Step 4/5: Aggregation"
echo "=============================================================="
echo "Aggregating all results (classifier + LLM scores)"
echo ""

# Run aggregation (creates results/z_classifier_scores/ and results/z_llm_scores/ directories)
PYTHONUNBUFFERED=1 python -m src.analysis.aggregation

# Check if aggregation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Aggregation failed"
    exit 1
fi

echo ""
echo "Aggregation completed successfully!"
echo ""

echo "Step 5/5: Turn-level Analysis & Visualization"
echo "=============================================================="
echo "Generating turn-level progression plots"
echo ""

# Find the turn-level CSV files and generate plots
TURN_CSV_DIR="results/z_llm_scores/turn_level"

# Only process specific CSV files (no_not_pt variants)
SPECIFIC_FILES=(
    "aggregated_llm_scores_by_turn_normal_no_not_pt.csv"
    "aggregated_llm_scores_by_turn_pt-br_no_not_pt.csv"
    "aggregated_llm_scores_by_turn_pt-pt_no_not_pt.csv"
)

if [ -d "$TURN_CSV_DIR" ]; then
    for csv_name in "${SPECIFIC_FILES[@]}"; do
        csv_file="$TURN_CSV_DIR/$csv_name"
        if [ -f "$csv_file" ]; then
            echo "Plotting: $csv_name"
            PYTHONUNBUFFERED=1 python -m src.analysis.turn_analysis "$csv_file" \
                --max-turns 3 \
                --no-show || echo "Warning: Failed to generate plot for $csv_file"
        else
            echo "Warning: File not found: $csv_name"
        fi
    done
else
    echo "Warning: Turn-level CSV directory not found: $TURN_CSV_DIR"
fi

echo ""
echo "Turn-level analysis completed!"
echo ""

echo "=============================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "=============================================================="
echo "Model: $MODEL"
echo "Output directory: $OUT_DIR"
echo ""
echo "Results summary:"
echo "  - Raw responses: $OUT_DIR/*.json"
echo "  - Classifier scores: $OUT_DIR/class_scores/*.csv"
echo "  - LLM judge scores: $OUT_DIR/llm_scores/*.csv"
echo "  - Aggregated classifier results: results/z_classifier_scores/"
echo "  - Aggregated LLM results: results/z_llm_scores/"
echo "  - Turn-level visualizations: results/z_llm_scores/turn_level/visualizations/"
echo "  - Combined comprehensive table: results/combined_comprehensive_scores_llm_scores.csv"
echo "=============================================================="
