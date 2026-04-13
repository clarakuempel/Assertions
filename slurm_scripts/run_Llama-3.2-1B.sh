#!/usr/bin/bash -l
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/score_meta-llama_Llama-3.2-1B_%j.out
#SBATCH --error=logs/score_meta-llama_Llama-3.2-1B_%j.err
#SBATCH --job-name=score_meta-llama_Llama-3.2-1B

# Load GPU and CUDA modules
module load t4 cuda/11.8.0 cudnn/8.7.0

module load mamba

source activate assertions

echo "Current environment: $(which python)"

pip install -r requirements.txt

# Display GPU information
nvidia-smi

python score_dataset.py --model_name "meta-llama/Llama-3.2-1B"

echo "GPU job completed successfully"
