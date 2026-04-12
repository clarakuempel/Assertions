import os
import sys
import subprocess

models = [
     "gpt-5-nano-2025-08-07",
     "gpt-5-mini-2025-08-07",
     "gpt-5-2025-08-07",
]

def main():
     script_path = os.path.join(os.path.dirname(__file__), "score_dataset.py")
     processes = []
     for model in models:
          # without --query_only
          cmd_default = [sys.executable, script_path, "-M", model]
          print(f"Starting: {' '.join(cmd_default)}")
          processes.append((model, "default", subprocess.Popen(cmd_default)))

          # with --query_only
          cmd_query_only = [sys.executable, script_path, "-M", model, "--query_only"]
          print(f"Starting: {' '.join(cmd_query_only)}")
          processes.append((model, "query_only", subprocess.Popen(cmd_query_only)))

     failed_runs = []
     for model, variant, process in processes:
          return_code = process.wait()
          if return_code != 0:
               failed_runs.append(f"{model} ({variant})")
               print(f"Model {model} [{variant}] failed with exit code {return_code}")
          else:
               print(f"Model {model} [{variant}] completed successfully")

     if failed_runs:
          print(f"One or more runs failed: {', '.join(failed_runs)}")
          sys.exit(1)

if __name__ == "__main__":
     main()

     