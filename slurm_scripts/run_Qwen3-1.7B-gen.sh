#!/usr/bin/bash -l
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu.4h
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_Qwen_Qwen3-1.7B_%j.out
#SBATCH --error=logs/score_Qwen_Qwen3-1.7B_%j.err
#SBATCH --job-name=score_Qwen_Qwen3-1.7B

cd /cluster/work/cotterell/kdu/Assertions

# Load GPU and CUDA modules
module load cuda/13.0.2

export MAMBA_EXE='/cluster/home/kevidu/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/cluster/work/cotterell/kdu/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"

micromamba activate /cluster/work/cotterell/kdu/envs/assertions

echo "Current environment: $(which python)"

pip install -r requirements.txt

# Display GPU information
nvidia-smi

python score_dataset.py --model_name "Qwen/Qwen3-1.7B" -I data/generated_assertions_v2_1000.jsonl --use_generate
 

echo "GPU job completed successfully"
