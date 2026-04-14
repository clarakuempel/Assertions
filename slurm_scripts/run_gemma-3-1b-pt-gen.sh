#!/usr/bin/bash -l
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu.4h
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_google_gemma-3-1b-pt_assertions_%j.out
#SBATCH --error=logs/score_google_gemma-3-1b-pt_assertions_%j.err
#SBATCH --job-name=score_google_gemma-3-1b-pt-assertions

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

python score_dataset.py --model_name "google/gemma-3-1b-pt" -I data/generated_assertions_v2_1000.jsonl --use_generate 

echo "GPU job completed successfully"
