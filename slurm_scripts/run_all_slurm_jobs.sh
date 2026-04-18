#!/usr/bin/env bash
# Submit every SBATCH driver in this directory (except this file).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

count=0
for script in run_*.sh; do
  if [[ "$script" == run_all_slurm_jobs.sh ]]; then
    continue
  fi
  sbatch "$script"
  count=$((count + 1))
done

echo "Submitted ${count} jobs."
