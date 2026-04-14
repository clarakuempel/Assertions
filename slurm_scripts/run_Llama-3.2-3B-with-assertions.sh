#!/usr/bin/bash -l
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu.4h
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_meta-llama_Llama-3.2-3B_assertions_%j.out
#SBATCH --error=logs/score_meta-llama_Llama-3.2-3B_assertions_%j.err
#SBATCH --job-name=score_meta-llama_Llama-3.2-3B-assertions

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

python score_dataset.py --model_name "meta-llama/Llama-3.2-3B" -I data/generated_assertions_v2_1000.jsonl --query_only

echo "GPU job completed successfully"
