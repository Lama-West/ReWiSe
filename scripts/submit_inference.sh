#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=def-azouaq
#SBATCH --job-name=submit_inference
#SBATCH --output=out_jobs/%x-%j.out
#SBATCH --error=out_jobs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

#
# SLURM script for submitting inference experiments
# This script runs the submit_inference.py script which generates configs and submits the actual inference jobs
#
# Usage:
#   sbatch scripts/submit_inference.sh --model l_8i --cot --dataset dataset2024
#   sbatch scripts/submit_inference.sh --trained-model-dir /path/to/peft/model
#

# Setup environment
source $HOME/.bashrc
source $HOME/myenv/bin/activate

# Navigate to project directory
cd /home/edarsem/projects/def-azouaq/edarsem/soft_knowledge_retrieval

# Create output directory if it doesn't exist
mkdir -p out_jobs
mkdir -p configs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Arguments: $@"
echo "=========================================="

# Run the submit_inference script with all provided arguments
python src/submit_inference.py "$@"

exit_code=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
