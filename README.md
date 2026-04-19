# Assertion Typology Framework

This repository contains code for studying how language models respond to factual assertions presented through different **Expressions of Belief (EoBs)**.
Given a false assertion followed by a Yes/No question about the true fact, we measure the **Context-Following Rate (CFR)** — how often the model follows the assertion rather than its memorized knowledge — across 19 linguistically motivated EoB categories.

## EoB Typology

The framework organizes EoBs into 4 dimensions with 19 total categories:

1. **Form** (7 categories): How the assertion is syntactically structured
   - `explicit` — direct declarative statements
   - `not_at_issue` — presupposition triggers (factive verbs, appositives, clefts)
   - `material_conditional` — if-then conditionals with true antecedent
   - `counterfactual` — subjunctive conditionals with false antecedent
   - `supposition` — hypothetical framing ("Suppose...", "Imagine...")
   - `imperative` — commands to accept information
   - `interrogative` — rhetorical questions implying the false fact

2. **Evidentiality** (3 categories): Claimed source of information
   - `hearsay` — attributed to unspecified others
   - `authority` — attributed to authoritative sources
   - `belief_reports` — attributed to a named believer

3. **Epistemic Stance** (2 categories): Expressed certainty
   - `strong` — high confidence markers
   - `weak` — hedged or uncertain framing

4. **Tone** (7 categories): Register and stylistic variation
   - `formal` — academic/professional register
   - `informal` — casual register
   - `poetic` — literary/metaphorical framing
   - `child_directed` — simplified, pedagogical register
   - `emotional_appeal` — affect-laden framing
   - `sarcasm` — ironic inversion
   - `social_media` — internet-native register

Each category has 10 templates defined in `preprocessing/assertion_templates.json`.

## Repository Structure

```
assertion_generator.py          # Generates assertion datasets from templates + facts
score_dataset.py                # Scores datasets with HuggingFace or OpenAI models
preprocessing/
  assertion_templates.json      # 19 categories x 10 templates each
  popqa.ipynb                   # PopQA preprocessing notebook
data/
  popqa_filtered_v2_enhanced.jsonl          # 3,462 PopQA-derived fact tuples
  generated_assertions_v2_{N}.jsonl         # Pre-generated datasets (N = 500, 1000, 2000, full)
slurm_scripts/                  # SBATCH scripts for cluster-based scoring
analysis/                       # Jupyter notebooks for results analysis
plots/                          # Generated figures for the paper
utils/                          # Shared utilities (model loading, IO, constants)
requirements.txt                # Python dependencies
```

## Setup

```bash
git clone <repo-url>
cd Assertions
pip install -r requirements.txt
```

For gated models (Llama, Gemma), you need a HuggingFace token:
```bash
huggingface-cli login
```

## Data Generation

Generate an assertion dataset with N sampled facts:

```bash
python assertion_generator.py -N 1000
```

This reads `data/popqa_filtered_v2_enhanced.jsonl` (3,462 facts from PopQA), samples N facts, and generates 19 categories x 2 query types x N rows.
Output is written to `data/generated_assertions_v2_{N}.jsonl`.

Each row contains an `assertion` (the EoB-formatted false claim), a `query` (Yes/No question about the true fact), `query_type` (`prior_yes` or `ctx_yes`), `dimension`, `category`, and the source `fact`.

Open-ended wh-questions (who / what / where from the fact’s `relation`) and `query_format=open_ended`:

```bash
python assertion_generator.py -N 1000 --open-questions
```

Writes `data/generated_assertions_v2_open_{N}.jsonl`. Score with `python score_dataset.py ... --use_generate` (required for local models).

## Scoring Models

Score a dataset with a HuggingFace model:

```bash
python score_dataset.py -M meta-llama/Llama-3.1-8B-Instruct -I data/generated_assertions_v2_full.jsonl
```

### Key flags

| Flag | Effect |
|---|---|
| `--query_only` | Prompt includes only the query (no assertion). Used as the baseline to measure CFR against. |
| `--use_generate` | Use greedy generation (10 tokens) instead of first-token Yes/No logit probabilities. Required for pretrained (non-instruct) models that don't reliably produce Yes/No as the first token. |

### Output

Results are saved to `data/{model_name}_{dataset_name}/`:
- `results.csv` (or `results_query_only.csv`) — per-example classification
- `summary.json` (or `summary_query_only.json`) — aggregate statistics

Each example is classified as:
- **memory** — model agrees with the true (memorized) fact
- **context** — model follows the false assertion
- **other** — response is not a clear Yes/No
- **error** — generation failed

## Running on a Cluster (Slurm)

The `slurm_scripts/` directory contains SBATCH scripts for all evaluated models.
Each script loads GPU and CUDA modules, activates the `assertions` conda environment, installs dependencies, and runs `score_dataset.py`.

### Script naming conventions

There are two drivers per HuggingFace checkpoint (18 models → 36 jobs). All use `data/generated_assertions_v2_full.jsonl` and `--use_generate`.

- `run_<MODEL>_query_only.sh` — query only (`--query_only --use_generate`); writes `results_query_only.csv` and `summary_query_only.json` under the model’s output directory.
- `run_<MODEL>_with_assertions.sh` — assertion then query (`--use_generate`); writes `results.csv` and `summary.json`.

### Submitting jobs

Submit a single model:
```bash
cd slurm_scripts
sbatch run_Llama-3.1-8B-Instruct_query_only.sh
```

Submit all scoring jobs (36):
```bash
cd slurm_scripts
bash run_all_slurm_jobs.sh
```

### Evaluated models

| Family | Models |
|---|---|
| Llama 3.1 | 8B, 8B-Instruct |
| Llama 3.2 | 1B, 1B-Instruct, 3B, 3B-Instruct |
| Gemma 3 | 1b-pt, 1b-it, 12b-pt, 12b-it, 27b-pt, 27b-it |
| Qwen 3 | 1.7B, 1.7B-Base, 8B, 8B-Base, 30B-A3B, 30B-A3B-Base |

## Analysis & Reproducing Paper Plots

All figures are generated from Jupyter notebooks in `analysis/`. Plots are saved to `plots/`.

### Primary notebook: `model_comparison_analysis_acl.ipynb`

This notebook produces the main paper figures:

| Plot file | Description |
|---|---|
| `query_only_status_stacked_by_model.png` | Baseline Yes/No response distribution per model |
| `context_following_rate_by_category_and_dimension_all_models.png` | Main CFR plot across all EoB categories |
| `context_following_rate_by_category_and_model_heatmap.png` | CFR heatmap (categories x models) |
| `context_following_rate_by_dimension_all_models_by_family.png` | CFR by dimension, grouped by model family |
| `context_following_rate_vs_size_lineplot_pretty.png` | CFR vs model size |
| `cfr_vs_size_fine_by_family.png` | Fine-grained CFR scaling by family |
| `context_following_rate_barplot_by_model_root_type.png` | Base vs instruct comparison |
| `context_following_rate_by_model_and_dimension_base_minus_instruct.png` | CFR delta (base minus instruct) |
| `spearman_correlation_of_context_pct_rankings_by_model.png` | Rank correlation across models |
| `rank_boxplot_sorted_dim_stars_left_new.png` | EoB category rank distributions |
| `per_model_dimension_bars_generic.png` | Per-model dimension bar charts |
| `dimension_heatmaps_generic_{base,instruct}_{absolute,delta}.png` | Dimension heatmaps by model type |

### Other notebooks

| Notebook | Purpose |
|---|---|
| `analyze_results2.ipynb` | Single-model CFR breakdown by category and dimension |
| `model_comparison_analysis.ipynb` | Earlier version of multi-model comparison |
| `dataset_analysis.ipynb` | Dataset statistics and distribution analysis |
| `answer_other.ipynb` | Analysis of non-Yes/No ("other") responses |

### Validation

Run assertion template validation tests:
```bash
python -m pytest analysis/test_generated_assertions.py -x -q
```
