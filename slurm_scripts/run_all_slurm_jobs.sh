#!/usr/bin/env bash
# Submit every SBATCH driver in this directory (except this file).
#
# Optional: point all jobs at another assertions JSONL (e.g. open-ended full set):
#   ASSERTIONS_JSONL=data/generated_assertions_v2_open_full.jsonl bash run_all_slurm_jobs.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

default_input="data/generated_assertions_v2_full.jsonl"
input_jsonl="${ASSERTIONS_JSONL:-$default_input}"

count=0
for script in run_*.sh; do
  if [[ "$script" == run_all_slurm_jobs.sh ]]; then
    continue
  fi
  if [[ "$input_jsonl" == "$default_input" ]]; then
    sbatch "$script"
  else
    sbatch <(sed "s|${default_input}|${input_jsonl}|g" "$script")
  fi
  count=$((count + 1))
done

echo "Submitted ${count} jobs (input JSONL: ${input_jsonl})."
