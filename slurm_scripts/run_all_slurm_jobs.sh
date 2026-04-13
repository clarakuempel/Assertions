#!/bin/bash

for script in \
  run_gemma-3-1b-pt-gen.sh\
  run_gemma-3-1b-pt-gen-quer.sh\
  run_Qwen3-1.7B-gen.sh\
  run_Qwen3-1.7B-gen-quer.sh\
  run_Qwen3-30B-A3B-gen.sh\
  run_Qwen3-30B-A3B-gen-quer.sh; do
  sbatch "$script"
done


