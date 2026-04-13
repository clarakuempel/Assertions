#!/usr/bin/bash -l
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/score_Qwen_Qwen3-30B-A3B-Base_%j.out
#SBATCH --error=logs/score_Qwen_Qwen3-30B-A3B-Base_%j.err
#SBATCH --job-name=score_Qwen_Qwen3-30B-A3B-Base

# Load GPU and CUDA modules
module load a100 cuda/11.8.0 cudnn/8.7.0

module load mamba

source activate assertions

echo "Current environment: $(which python)"

pip install -r requirements.txt

# Display GPU information
nvidia-smi

python score_dataset.py --model_name "Qwen/Qwen3-30B-A3B-Base"

echo "GPU job completed successfully"
