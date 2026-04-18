#!/usr/bin/bash -l
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cuda13pr.24h
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_gemma-3-27b-pt_query_only_%j.out
#SBATCH --error=logs/score_gemma-3-27b-pt_query_only_%j.err
#SBATCH --job-name=score_gemma-3-27b-pt_query_only
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

python score_dataset.py --model_name "google/gemma-3-27b-pt" -I data/generated_assertions_v2_full.jsonl --query_only --use_generate

echo "GPU job completed successfully"
