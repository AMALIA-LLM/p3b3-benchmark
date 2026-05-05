#!/bin/bash -l


# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=p3b3_llm_scoring
#SBATCH --array=0
#SBATCH --output=./joblog/%x-%A_%a.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0-11:00:00
#SBATCH --partition=high_priority


eval "$(conda shell.bash hook)"

# activate conda environment
conda activate p3b3

# go to folder
cd "$HOME/P3B3"


# Array of directories to process
DIRECTORIES=(
#    "$HOME/P3B3/results/allenai-Olmo-3-7B-Instruct"
#    "$HOME/P3B3/results/allenai-Olmo-3.1-32B-Instruct"
#
#    "$HOME/P3B3/results/BSC-LT-salamandra-7b-instruct"
#
#    "$HOME/P3B3/results/swiss-ai-Apertus-70B-Instruct-2509"
#    "$HOME/P3B3/results/swiss-ai-Apertus-8B-Instruct-2509"
#
#    "$HOME/P3B3/results/utter-project-EuroLLM-22B-Instruct-2512"
#    "$HOME/P3B3/results/utter-project-EuroLLM-9B-Instruct-2512"
#
#    "$HOME/P3B3/results/carminho-AMALIA-9B-50-1225-DPO"
#    "$HOME/P3B3/results/carminho-AMALIA-9B-50-1225-SFT"
#
#    "$HOME/P3B3/results/google-gemma-3-12b-it"
#    "$HOME/P3B3/results/google-gemma-3-27b-it"
#    "$HOME/P3B3/results/google-gemma-4-31B-it"
#    "$HOME/P3B3/results/google-gemma-4-E4B-it"
#
#    "$HOME/P3B3/results/meta-llama-Llama-3.1-8B-Instruct"
#    "$HOME/P3B3/results/meta-llama-Llama-3.3-70B-Instruct"
#
#    "$HOME/P3B3/results/PORTULAN-gervasio-8b-portuguese-ptpt-decoder"
#    "$HOME/P3B3/results/PORTULAN-gervasio-70b-portuguese-ptpt-decoder"
#
#    "$HOME/P3B3/results/mistralai-Ministral-3-14B-Instruct-2512"
#    #"$HOME/P3B3/results/mistralai-Ministral-3-8B-Instruct-2512"  # broken in vllm
#
#    "$HOME/P3B3/results/Qwen-Qwen3-8B"
#    "$HOME/P3B3/results/Qwen-Qwen3.5-27B"
#    "$HOME/P3B3/results/Qwen-Qwen3.5-9B"
#
#    # closed source
#    "$HOME/P3B3/results/maritaca-api-sabia-4"
#    "$HOME/P3B3/results/maritaca-api-sabia-3.1"
#
#    "$HOME/P3B3/results/google-langchain-api-gemini-3-flash-preview"

)

# Loop over all directories
for DIR_TO_USE in "${DIRECTORIES[@]}"; do
    echo "========================================"
    echo "Starting LLM scoring for directory: $DIR_TO_USE"
    echo "========================================"

    PYTHONUNBUFFERED=1 python -m src.evaluation.score_with_llm "$DIR_TO_USE" --max_connections 50 --no-accumulate-context

    echo "Completed: $DIR_TO_USE"
    echo ""
done

echo "All directories processed!"
