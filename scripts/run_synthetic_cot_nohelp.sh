#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-azouaq
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=synthetic_cot_nohelp
#SBATCH --output=out_jobs/%x-%j.out

# Parse command-line arguments
TEMP=${1:-1.0}      # Default temperature is 1.0
FEW_SHOT=${2:-2}    # Default few-shot is 2 (same as Lambda API)
REPEATS=${3:-10}    # Default is 10 repeats of the full dataset (same as Lambda API)
MODEL=${4:-l_8i}    # Default model is l_8i (Llama-8B)
GPUS=${5:-2}        # Default number of GPUs is 2

# Update job name to include parameters
#SBATCH --job-name=synthetic_cot_nohelp_${FEW_SHOT}shot_temp${TEMP}

# Format output filename based on parameters
OUTPUT_FILE="data/dataset2025/cot/nohelp_synthetic_cot/raw/synthetic_${MODEL}.csv"
CLEAN_OUTPUT_FILE="data/dataset2025/cot/nohelp_synthetic_cot/clean/synthetic_${MODEL}_clean.csv"

# Fail immediately if a command fails
set -e

# Activate the virtual environment
source $HOME/.bashrc
source $HOME/myenv/bin/activate

# Navigate to project directory
cd /home/edarsem/projects/def-azouaq/edarsem/soft_knowledge_retrieval

# Copy model to SLURM_TMPDIR for faster access
if [ -n "$SLURM_TMPDIR" ]; then
    chmod +x scripts/copy_model.sh
    ./scripts/copy_model.sh $MODEL quantized
    # Create temp config with SLURM_TMPDIR as a model directory
    export CONFIG_INI=$SLURM_TMPDIR/config.ini
fi

# Display run parameters
echo "Running synthetic CoT generation with parameters:"
echo "  Model: $MODEL"
echo "  Temperature: $TEMP"
echo "  Few-shot examples: $FEW_SHOT"
echo "  Dataset repeats: $REPEATS"
echo "  Number of GPUs: $GPUS"
echo "  Output file: $OUTPUT_FILE"

# Run the synthetic COT generation with model aliases from config - "nohelp" mode
python src/synthetic_cot.py --model $MODEL --vllm --quantized --few-shot $FEW_SHOT --all-relations --max-tokens 1000 \
    --full-dataset-repeats $REPEATS --output $OUTPUT_FILE --temperature $TEMP \
    --mode nohelp --dataset dataset2025 --tensor-parallel-size $GPUS

# Run evaluation on the generated data
python src/process_synthetic_cot.py --input $OUTPUT_FILE \
    --output $CLEAN_OUTPUT_FILE --mode nohelp --dataset dataset2025

echo "Job completed at $(date)"
