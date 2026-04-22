from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemplatePattern:
    category: str
    template_index: int
    template_text: str
    regex: re.Pattern[str]


def _compile_template_regex(template: str) -> re.Pattern[str]:
    escaped = re.escape(template)
    wildcarded = re.sub(r"\\\{[^{}]+\\\}", r".+?", escaped)
    return re.compile(f"^{wildcarded}$")


def load_template_patterns(template_path: Path) -> tuple[dict[str, list[TemplatePattern]], list[str]]:
    with template_path.open("r", encoding="utf-8") as f:
        template_data = json.load(f)

    patterns_by_category: dict[str, list[TemplatePattern]] = {}
    category_order: list[str] = []

    for _dimension, categories in template_data.items():
        for category, payload in categories.items():
            templates: list[str] = payload["templates"]
            compiled = [
                TemplatePattern(
                    category=category,
                    template_index=template_index,
                    template_text=template,
                    regex=_compile_template_regex(template),
                )
                for template_index, template in enumerate(templates)
            ]
            patterns_by_category[category] = compiled
            category_order.append(category)

    return patterns_by_category, category_order


def infer_template_index(assertion: str, category: str, patterns_by_category: dict[str, list[TemplatePattern]]) -> int | None:
    for pattern in patterns_by_category[category]:
        if pattern.regex.fullmatch(assertion):
            return pattern.template_index
    return None


def collect_template_level_cfr(
    data_dir: Path, patterns_by_category: dict[str, list[TemplatePattern]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_paths = sorted(data_dir.glob("*/results.csv"))
    if not results_paths:
        raise FileNotFoundError(f"No results.csv files found under {data_dir}")

    model_level_frames: list[pd.DataFrame] = []
    unmatched_rows: list[dict[str, str | int]] = []

    for results_path in results_paths:
        model_name = results_path.parent.name
        df = pd.read_csv(results_path, usecols=["assertion", "category", "classification"])
        df = df[df["category"].isin(patterns_by_category)].copy()
        df["template_index"] = [
            infer_template_index(assertion=a, category=c, patterns_by_category=patterns_by_category)
            for a, c in zip(df["assertion"], df["category"])
        ]

        unmatched = df[df["template_index"].isna()]
        if not unmatched.empty:
            unmatched_rows.append(
                {
                    "model_name": model_name,
                    "unmatched_count": int(unmatched.shape[0]),
                }
            )

        matched = df.dropna(subset=["template_index"]).copy()
        matched["template_index"] = matched["template_index"].astype(int)
        matched["is_context"] = (matched["classification"] == "context").astype(float)
        matched["model_name"] = model_name
        model_level_frames.append(matched)

    all_rows = pd.concat(model_level_frames, ignore_index=True)
    template_cfr = (
        all_rows.groupby(["model_name", "category", "template_index"], as_index=False)["is_context"]
        .mean()
        .rename(columns={"is_context": "cfr"})
    )

    unmatched_df = pd.DataFrame(unmatched_rows)
    return template_cfr, unmatched_df


def summarize_observed_variance(template_cfr: pd.DataFrame, output_table_path: Path) -> pd.DataFrame:
    variance_df = (
        template_cfr.groupby(["model_name", "category"], as_index=False)["cfr"]
        .var(ddof=0)
        .rename(columns={"cfr": "template_cfr_variance"})
    )
    variance_df.to_csv(output_table_path, index=False)
    return variance_df


def summarize_permutation_baseline(
    template_cfr: pd.DataFrame,
    category_order: list[str],
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    baseline_rows: list[dict[str, str | int | float]] = []

    for model_name, model_df in template_cfr.groupby("model_name"):
        model_df = model_df.sort_values(["category", "template_index"]).reset_index(drop=True)
        values = model_df["cfr"].to_numpy()
        labels = model_df["category"].to_numpy()

        for permutation_idx in range(n_permutations):
            shuffled_values = rng.permutation(values)
            shuffled_df = pd.DataFrame({"category": labels, "cfr": shuffled_values})
            perm_variance = shuffled_df.groupby("category", as_index=False)["cfr"].var(ddof=0)
            perm_variance["model_name"] = model_name
            perm_variance["permutation_idx"] = permutation_idx
            baseline_rows.extend(
                {
                    "model_name": row["model_name"],
                    "category": row["category"],
                    "permutation_idx": int(row["permutation_idx"]),
                    "template_cfr_variance": float(row["cfr"]),
                }
                for _, row in perm_variance.iterrows()
            )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df["category"] = pd.Categorical(baseline_df["category"], categories=category_order, ordered=True)
    return baseline_df


def build_observed_only_plot(
    observed_variance_df: pd.DataFrame, category_order: list[str], output_plot_path: Path
) -> None:

    summary_df = (
        observed_variance_df.groupby("category", as_index=False)["template_cfr_variance"]
        .agg(mean_variance="mean", std_variance="std", model_count="count")
    )
    summary_df["std_variance"] = summary_df["std_variance"].fillna(0.0)
    summary_df["category"] = pd.Categorical(summary_df["category"], categories=category_order, ordered=True)
    summary_df = summary_df.sort_values("category")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(summary_df["category"], summary_df["mean_variance"], color="#4C72B0", alpha=0.9)
    ax.errorbar(
        summary_df["category"],
        summary_df["mean_variance"],
        yerr=summary_df["std_variance"],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax.set_ylabel("Variance of template-level CFR (across templates)")
    ax.set_xlabel("Assertion type (category)")
    ax.set_title("CFR variance across templates within each assertion type")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=300)
    plt.close(fig)


def build_observed_vs_permutation_plot(
    observed_variance_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    category_order: list[str],
    output_plot_path: Path,
) -> None:
    observed_summary = (
        observed_variance_df.groupby("category", as_index=False)["template_cfr_variance"]
        .agg(observed_mean="mean", observed_std="std")
    )
    observed_summary["observed_std"] = observed_summary["observed_std"].fillna(0.0)

    baseline_summary = (
        baseline_df.groupby("category", as_index=False, observed=False)["template_cfr_variance"]
        .agg(baseline_mean="mean", baseline_std="std")
    )
    baseline_summary["baseline_std"] = baseline_summary["baseline_std"].fillna(0.0)

    merged = observed_summary.merge(baseline_summary, on="category", how="inner")
    merged["category"] = pd.Categorical(merged["category"], categories=category_order, ordered=True)
    merged = merged.sort_values("category")

    x = np.arange(len(merged))
    width = 0.42

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(
        x - width / 2,
        merged["observed_mean"],
        width=width,
        color="#4C72B0",
        yerr=merged["observed_std"],
        capsize=3,
        label="Observed",
    )
    ax.bar(
        x + width / 2,
        merged["baseline_mean"],
        width=width,
        color="#DD8452",
        yerr=merged["baseline_std"],
        capsize=3,
        label="Permutation baseline",
    )
    ax.set_ylabel("Variance of template-level CFR (across templates)")
    ax.set_xlabel("Assertion type (category)")
    ax.set_title("Observed CFR variance vs random template-permutation baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["category"])
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=300)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_path = repo_root / "preprocessing" / "assertion_templates.json"
    data_dir = repo_root / "data"
    output_plot_path = repo_root / "plots" / "cfr_template_variance_by_category.png"
    output_table_path = repo_root / "comparison_results" / "cfr_template_variance_by_model_and_category.csv"
    output_baseline_plot_path = repo_root / "plots" / "cfr_template_variance_with_permutation_baseline.png"
    output_baseline_table_path = repo_root / "comparison_results" / "cfr_template_variance_permutation_baseline.csv"

    patterns_by_category, category_order = load_template_patterns(template_path=template_path)
    template_cfr, unmatched_df = collect_template_level_cfr(data_dir=data_dir, patterns_by_category=patterns_by_category)
    observed_variance_df = summarize_observed_variance(
        template_cfr=template_cfr,
        output_table_path=output_table_path,
    )
    build_observed_only_plot(
        observed_variance_df=observed_variance_df,
        category_order=category_order,
        output_plot_path=output_plot_path,
    )
    baseline_df = summarize_permutation_baseline(template_cfr=template_cfr, category_order=category_order)
    baseline_df.to_csv(output_baseline_table_path, index=False)
    build_observed_vs_permutation_plot(
        observed_variance_df=observed_variance_df,
        baseline_df=baseline_df,
        category_order=category_order,
        output_plot_path=output_baseline_plot_path,
    )

    if not unmatched_df.empty:
        print("Warning: Some assertions did not match a template pattern:")
        print(unmatched_df.to_string(index=False))
    else:
        print("All assertions matched a template pattern.")

    print(f"Wrote plot: {output_plot_path}")
    print(f"Wrote per-model variance table: {output_table_path}")
    print(f"Wrote observed-vs-baseline plot: {output_baseline_plot_path}")
    print(f"Wrote permutation baseline table: {output_baseline_table_path}")


if __name__ == "__main__":
    main()
