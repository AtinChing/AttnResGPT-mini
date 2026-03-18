from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_probe_series(run_dir: Path) -> list[dict]:
    probes = []
    for probe_path in sorted((run_dir / "probes").glob("step_*.pt")):
        probes.append(torch.load(probe_path, map_location="cpu"))
    return probes


def plot_loss_curves(run_dirs: list[Path], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_dir in run_dirs:
        train_df = load_jsonl(run_dir / "train_metrics.jsonl")
        val_df = load_jsonl(run_dir / "val_metrics.jsonl")
        if not train_df.empty:
            plt.plot(train_df["step"], train_df["train_loss"], label=f"{run_dir.name} train")
        if not val_df.empty:
            plt.plot(val_df["step"], val_df["val_loss"], marker="o", label=f"{run_dir.name} val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=200)
    plt.close()


def _latest_probe(run_dir: Path) -> dict | None:
    probes = load_probe_series(run_dir)
    return probes[-1] if probes else None


def plot_layer_norms(run_dirs: list[Path], output_dir: Path, key: str, filename: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    for run_dir in run_dirs:
        probe = _latest_probe(run_dir)
        if probe is None:
            continue
        values = probe.get(key, {})
        if not values:
            continue
        labels = list(values.keys())
        magnitudes = [values[label] for label in labels]
        plt.plot(range(len(labels)), magnitudes, marker="o", label=run_dir.name)
    plt.xlabel("layer / module index")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def plot_depth_attention_heatmaps(run_dirs: list[Path], output_dir: Path) -> None:
    for run_dir in run_dirs:
        probe = _latest_probe(run_dir)
        if probe is None or not probe.get("depth_attention"):
            continue
        rows = []
        for row in probe["depth_attention"]:
            rows.append(row["mean_weights"])
        width = max(len(row) for row in rows)
        matrix = []
        for row in rows:
            matrix.append(row + [0.0] * (width - len(row)))
        plt.figure(figsize=(10, 5))
        sns.heatmap(matrix, cmap="mako", cbar=True)
        plt.xlabel("source index within available history")
        plt.ylabel("depth attention row")
        plt.title(f"Depth attention heatmap: {run_dir.name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"depth_attention_{run_dir.name}.png", dpi=200)
        plt.close()


def plot_contribution_summary(run_dirs: list[Path], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_dir in run_dirs:
        probes = load_probe_series(run_dir)
        if not probes:
            continue
        steps = []
        early = []
        late = []
        embedding = []
        for probe in probes:
            scalar_aux = probe.get("scalar_aux", {})
            if "early_contribution" not in scalar_aux:
                continue
            steps.append(probe["step"])
            early.append(scalar_aux["early_contribution"])
            late.append(scalar_aux["late_contribution"])
            embedding.append(scalar_aux["embedding_contribution"])
        if not steps:
            continue
        plt.plot(steps, early, label=f"{run_dir.name} early")
        plt.plot(steps, late, label=f"{run_dir.name} late")
        plt.plot(steps, embedding, label=f"{run_dir.name} embed")
    plt.xlabel("step")
    plt.ylabel("mean contribution")
    plt.title("Depth contribution summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "contribution_summary.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved metrics and probe files.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to plot.")
    parser.add_argument("--output-dir", required=True, help="Directory for plot images.")
    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.run_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(run_dirs, output_dir)
    plot_layer_norms(run_dirs, output_dir, "gradient_norms", "gradient_norms.png", "gradient norms")
    plot_layer_norms(run_dirs, output_dir, "activation_norms", "activation_norms.png", "activation norms")
    plot_depth_attention_heatmaps(run_dirs, output_dir)
    plot_contribution_summary(run_dirs, output_dir)


if __name__ == "__main__":
    main()
