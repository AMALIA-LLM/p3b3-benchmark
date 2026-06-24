#!/bin/bash

# Script to submit all models as separate SLURM jobs
# Each model will be generated to results/<model_name>/ and evaluated immediately after
# Usage: ./submit_all_models.sh

echo "Submitting generation + evaluation jobs for all models"
echo "Each model will be saved to: results/<model_name>/"
echo "=============================================================="

MODELS=(

    #"allenai/Olmo-3-7B-Instruct"
    #"allenai/Olmo-3.1-32B-Instruct"

    #"BSC-LT/salamandra-7b-instruct"
    #"swiss-ai/Apertus-8B-Instruct-2509"
    #"swiss-ai/Apertus-70B-Instruct-2509"

    #"utter-project/EuroLLM-9B-Instruct-2512"
    #"utter-project/EuroLLM-22B-Instruct-2512"

    #"amalia-llm/AMALIA-9B-50-1225-SFT"
    #"amalia-llm/AMALIA-9B-50-1225-DPO"

    #"meta-llama/Llama-3.1-8B-Instruct"
    #"meta-llama/Llama-3.3-70B-Instruct"

    #"PORTULAN/gervasio-8b-portuguese-ptpt-decoder"
    #"PORTULAN/gervasio-70b-portuguese-ptpt-decoder"

    #"google/gemma-3-12b-it"
    #"google/gemma-3-27b-it"
    #"google/gemma-4-E4B-it"
    #"google/gemma-4-31B-it"

    #"Qwen/Qwen3-8B"
    #"Qwen/Qwen3.5-9B"
    #"Qwen/Qwen3.5-27B"

    # "mistralai/Ministral-3-8B-Instruct-2512"  # this is broken in vllm
    # "mistralai/Ministral-3-14B-Instruct-2512"

    # closed source
    #"google-langchain-api/gemini-3-flash-preview"

    #"maritaca-api/sabia-3.1"
    #"maritaca-api/sabia-4"

)

# Declare associative array for models that need another number of GPUs
# Format: ["model/name"]="gpu_count"
declare -A MODEL_GPU_REQUIREMENTS=(
    # Add more models here as needed
    ["allenai/Olmo-3.1-32B-Instruct"]="2"
    ["utter-project/EuroLLM-22B-Instruct-2512"]="2"
    ["google/gemma-4-31B-it"]="2"
    ["google/gemma-3-27b-it"]="2"
    ["Qwen/Qwen3.5-27B"]="2"
    ["meta-llama/Llama-3.3-70B-Instruct"]="2"
    ["PORTULAN/gervasio-70b-portuguese-ptpt-decoder"]="2"
    ["swiss-ai/Apertus-70B-Instruct-2509"]="2"

    # Gemini and sabia does not use gpu since it uses API
    ["google-langchain-api/gemini-3-flash-preview"]="0"
    ["maritaca-api/sabia-3.1"]="0"
    ["maritaca-api/sabia-4"]="0"
)


# Array to store job IDs
JOB_IDS=()

# Submit each model as a separate job
for MODEL in "${MODELS[@]}"; do
    # Extract a clean model name for the job name (replace / with -)
    MODEL_NAME_CLEAN=$(echo "$MODEL" | sed 's/\//-/g')

    # Get GPU count for this model (default to 1 if not specified)
    GPU_COUNT="${MODEL_GPU_REQUIREMENTS[$MODEL]:-1}"

    echo "Submitting job for: $MODEL (GPUs: $GPU_COUNT)"

    # Submit the job and capture the job ID
    JOB_ID=$(sbatch --job-name="p3b3_gen_${MODEL_NAME_CLEAN}" \
                    --gres=gpu:$GPU_COUNT \
                    run_scripts/run_single_model.sh "$MODEL" "$GPU_COUNT" | awk '{print $4}')

    JOB_IDS+=($JOB_ID)
    echo "  -> Job ID: $JOB_ID (results/${MODEL_NAME_CLEAN}/)"
done

echo ""
echo "=============================================================="
echo "All jobs submitted! Total: ${#JOB_IDS[@]} jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Each job will:"
echo "  1. Generate outputs to results/<model_name>/"
echo "  2. Automatically run evaluation/scoring"
echo ""
echo "To monitor jobs, run: squeue -u \$USER"
echo "To cancel all jobs, run: scancel ${JOB_IDS[*]}"
echo ""
echo "Results will be in separate directories under results/"
