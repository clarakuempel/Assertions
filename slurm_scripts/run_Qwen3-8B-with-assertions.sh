#!/usr/bin/bash -l
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/score_Qwen_Qwen3-8B_assertions_%j.out
#SBATCH --error=logs/score_Qwen_Qwen3-8B_assertions_%j.err
#SBATCH --job-name=score_Qwen_Qwen3-8B-assertions

# Load GPU and CUDA modules
module load a100 cuda/11.8.0 cudnn/8.7.0

module load mamba

source activate assertions

echo "Current environment: $(which python)"

pip install -r requirements.txt

# Display GPU information
nvidia-smi

python score_dataset.py --model_name "Qwen/Qwen3-8B" -I data/generated_assertions_v2_1000.jsonl --query_only

echo "GPU job completed successfully"
