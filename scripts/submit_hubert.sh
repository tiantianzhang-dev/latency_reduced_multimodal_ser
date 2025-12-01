#!/bin/bash
#SBATCH --job-name=hubert_ser
#SBATCH --output=hubert_%j.out
#SBATCH --error=hubert_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

# SLURM job submission script for Speech Emotion Recognition training
# Usage: sbatch scripts/submit_hubert.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules (adjust for your cluster)
module load python/3.8
module load cuda/11.8
module load cudnn/8.0

# Activate virtual environment
source venv/bin/activate

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Navigate to source directory
cd src

# Run training
echo "Starting training..."
python train.py

echo "Job completed at: $(date)"
