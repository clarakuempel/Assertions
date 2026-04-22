#!/usr/bin/env bash
# Submit every Gemma SBATCH driver in this directory.
#
# Optional: point all jobs at another assertions JSONL (e.g. open-ended full set):
#   ASSERTIONS_JSONL=data/generated_assertions_v2_open_full.jsonl bash run_all_gemma_slurm_jobs.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

default_input="data/generated_assertions_v2_full.jsonl"
input_jsonl="${ASSERTIONS_JSONL:-$default_input}"

shopt -s nullglob
gemma_scripts=(run_gemma-*.sh)

if [[ ${#gemma_scripts[@]} -eq 0 ]]; then
  echo "No Gemma scripts found (expected run_gemma-*.sh in ${script_dir})."
  exit 1
fi

count=0
for script in "${gemma_scripts[@]}"; do
  if [[ "$input_jsonl" == "$default_input" ]]; then
    sbatch "$script"
  else
    sbatch <(sed "s|${default_input}|${input_jsonl}|g" "$script")
  fi
  count=$((count + 1))
done

echo "Submitted ${count} Gemma jobs (input JSONL: ${input_jsonl})."
