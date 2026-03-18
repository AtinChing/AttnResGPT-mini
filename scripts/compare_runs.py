from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two completed experiment runs.")
    parser.add_argument("--baseline-run", required=True, help="Path to a baseline run directory.")
    parser.add_argument("--attnres-run", required=True, help="Path to an AttnRes run directory.")
    args = parser.parse_args()

    baseline_run = Path(args.baseline_run)
    attnres_run = Path(args.attnres_run)
    baseline = load_summary(baseline_run)
    attnres = load_summary(attnres_run)

    param_delta_pct = 100.0 * (
        (attnres["parameter_count_total"] - baseline["parameter_count_total"])
        / max(1, baseline["parameter_count_total"])
    )
    val_loss_delta = baseline["val_loss"] - attnres["val_loss"]
    ppl_delta = baseline["val_perplexity"] - attnres["val_perplexity"]

    rows = [
        ("baseline_run", str(baseline_run)),
        ("attnres_run", str(attnres_run)),
        ("baseline_params", baseline["parameter_count_total"]),
        ("attnres_params", attnres["parameter_count_total"]),
        ("parameter_delta_pct", param_delta_pct),
        ("baseline_val_loss", baseline["val_loss"]),
        ("attnres_val_loss", attnres["val_loss"]),
        ("val_loss_improvement", val_loss_delta),
        ("baseline_val_perplexity", baseline["val_perplexity"]),
        ("attnres_val_perplexity", attnres["val_perplexity"]),
        ("perplexity_improvement", ppl_delta),
        ("baseline_best_val_loss", baseline.get("best_val_loss")),
        ("attnres_best_val_loss", attnres.get("best_val_loss")),
        ("baseline_early_ablation_25", baseline.get("ablation_early_25_loss")),
        ("attnres_early_ablation_25", attnres.get("ablation_early_25_loss")),
        ("baseline_late_ablation_25", baseline.get("ablation_late_25_loss")),
        ("attnres_late_ablation_25", attnres.get("ablation_late_25_loss")),
        ("attnres_embedding_contribution", attnres.get("embedding_contribution")),
        ("attnres_early_contribution", attnres.get("early_contribution")),
        ("attnres_late_contribution", attnres.get("late_contribution")),
        ("attnres_depth_attention_entropy", attnres.get("depth_attention_entropy")),
    ]

    print("metric,value")
    for key, value in rows:
        print(f"{key},{format_metric(value)}")


if __name__ == "__main__":
    main()
