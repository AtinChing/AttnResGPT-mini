# AttnResGPT-mini

Small, runnable PyTorch research code for controlled experiments comparing standard residual connections against Attention Residuals (AttnRes) in a decoder-only GPT-style language model.

## Research Question

Primary question:

Does Attention Residuals make early layers more useful, and does it improve depth utilization compared to standard residual connections?

Secondary questions:

- Does AttnRes improve gradient flow across depth?
- Does it reduce activation norm growth relative to standard PreNorm residual streams?
- Do learned depth-attention weights reveal broader and more selective depth usage?

## What Is Implemented

### Baseline

The baseline block follows the usual PreNorm GPT pattern:

```text
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

### AttnRes

This repo replaces fixed residual accumulation with depth-wise softmax attention over previous layer outputs.

At small scale, each attention sublayer and each MLP sublayer gets its own learned pseudo-query vector. Source 0 is the token-plus-position embedding, and each subsequent source is the output of one previous sublayer. The depth mixer is:

```text
h_l = sum_i alpha_{i->l} * v_i
alpha_{i->l} = softmax_i( w_l^T RMSNorm(v_i) / temperature )
```

Supported variants:

- Full depth attention over all previous sources
- Sliding-window depth attention, optionally keeping source 0
- Optional final depth-attention readout before the LM head

## Paper Alignment

Primary source of truth:

- `2603.15031v1.pdf` in this repo

Public reference repo consulted as a spec:

- [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

Paper details reflected directly in this implementation:

- Learned pseudo-query vector per depth-mixing site
- RMSNorm on keys before computing depth-attention logits
- Zero initialization of depth queries
- Embedding included as source 0
- Separate pre-attention and pre-MLP depth mixing
- Training diagnostics centered on loss, norm growth, gradients, and learned depth weights

## Faithfulness vs Simplification

This repo is intentionally faithful to the paper's core mechanism, but not an exact reproduction of MoonshotAI's large-scale system.

Included:

- Decoder-only GPT-style causal LM
- Full AttnRes over sublayer outputs
- Sliding-window AttnRes as a practical small-scale ablation
- Fair training loop and matched configs
- Gradient and activation probes
- Early-vs-late ablation tooling

Deliberately simplified:

- No Kimi Linear architecture
- No MoE layers
- No pipeline-parallel communication or cache optimizations
- No large-scale Block AttnRes systems engineering stack
- No claim of exact large-model reproduction

These simplifications are intentional because the target environment is a single Colab T4 and the goal is controlled depth-utilization research, not industrial-scale reproduction.

## Repo Layout

```text
README.md
requirements.txt
configs/
src/
scripts/
notebooks/
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Debug run:

```bash
python -m src.train --config configs/debug_tiny.yaml
```

Baseline T4-small run:

```bash
python -m src.train --config configs/baseline_t4_small.yaml
```

AttnRes T4-small run:

```bash
python -m src.train --config configs/attnres_t4_small.yaml
```

Short matched pilot comparison:

```bash
bash scripts/run_pilot.sh
```

## Data Options

Supported datasets:

- `synthetic`: generated patterned text for debugging, overfit checks, and short pilots
- `text`: a local text file or folder of `.txt`, `.json`, or `.jsonl`
- `tinystories`: same loader path as `text`, intended for local TinyStories-style corpora

Examples:

```bash
python -m src.train \
  --config configs/baseline_t4_small.yaml \
  --overrides data.dataset_type=text data.text_path=/path/to/corpus.txt
```

```bash
python -m src.train \
  --config configs/attnres_t4_small.yaml \
  --overrides data.dataset_type=tinystories data.text_path=/path/to/tinystories_dir
```

## Experiment Workflow

1. Run sanity tests.

```bash
pytest -q tests/test_shapes.py tests/test_masks.py tests/test_forward_pass.py tests/test_attnres.py
```

2. Run a tiny overfit check.

```bash
pytest -q tests/test_tiny_overfit.py -m slow
```

3. Run a 50-step smoke test with matched overrides.

```bash
python -m src.train \
  --config configs/pilot_t4.yaml \
  --overrides experiment.name=smoke_baseline model.architecture=baseline training.max_steps=50

python -m src.train \
  --config configs/pilot_t4.yaml \
  --overrides experiment.name=smoke_attnres model.architecture=attnres model.attnres.enabled=true training.max_steps=50
```

4. Run the 500-1000 step pilot comparison.

```bash
bash scripts/run_pilot.sh
```

5. Compare runs and make plots.

```bash
python scripts/compare_runs.py \
  --baseline-run runs/pilot_baseline_<timestamp> \
  --attnres-run runs/pilot_attnres_<timestamp>

python scripts/plot_metrics.py \
  --run-dirs runs/pilot_baseline_<timestamp> runs/pilot_attnres_<timestamp> \
  --output-dir plots/pilot_compare
```

## What Gets Logged

Per-step or periodic logs:

- train loss
- val loss
- perplexity
- learning rate
- global gradient norm
- activation norms per layer module
- gradient norms per layer module
- mean depth-attention weights
- embedding / early / late contribution summaries
- ablation losses for early-vs-late layer removal

Artifacts saved under each run directory:

- `resolved_config.yaml`
- `train_metrics.jsonl`
- `val_metrics.jsonl`
- `run_summary.json`
- `checkpoints/`
- `probes/`
- `tokenizer.json`

## Colab Workflow

Five Colab notebooks are included:

- `notebooks/1_debug_and_sanity.ipynb`
- `notebooks/2_tiny_overfit.ipynb`
- `notebooks/3_smoke_test.ipynb`
- `notebooks/4_pilot_comparison.ipynb`
- `notebooks/5_analysis_and_plots.ipynb`

Each notebook:

- installs dependencies in the first runnable setup cell
- auto-clones the repo from `https://github.com/AtinChing/AttnResGPT-mini.git` into `/content/AttnResGPT-mini` if it is not already present
- mounts Drive and searches standard repo locations
- includes a GPU check cell
- runs the relevant tests or scripts sequentially
- produces plots or textual summaries without manual code edits

## Plotting Guide

The plotting script generates:

- `loss_curves.png`
- `gradient_norms.png`
- `activation_norms.png`
- `depth_attention_<run_name>.png`
- `contribution_summary.png`

Example:

```bash
python scripts/plot_metrics.py \
  --run-dirs runs/baseline_t4_small_<timestamp> runs/attnres_t4_small_<timestamp> \
  --output-dir plots/main_compare
```

## Design Choices and Assumptions

- The paper treats attention and MLP sublayers as separate depth steps. This repo does the same.
- The final readout uses an AttnRes aggregation by default because the paper states the final output layer aggregates prior sources.
- Sliding-window AttnRes is included as a principled small-scale ablation similar in spirit to the paper's limited-access ablations.
- Parameter matching is approximate rather than exact; the code logs the percent difference, which should stay small because AttnRes adds only lightweight query vectors and RMSNorms.
- Tokenization is intentionally simple and local-first to avoid external dependencies and to keep Colab runs robust.

## Potential Pitfalls

- Full depth attention grows linearly in stored source states and quadratically in depth mixing sites, so very deep runs can become memory-heavy even on small models.
- Synthetic data is useful for mechanistic comparison, but conclusions should be validated on real text before claiming broad language-model benefits.
- Tiny overfit success does not imply better depth utilization; use gradient, activation, and ablation probes together.
- AttnRes query zero-init is important. Random query init can make short runs unstable.
- Baseline and AttnRes should always be compared with the same seed, dataset, optimizer, and total steps.

## Next Research Extensions

- Add a true Block AttnRes small-model variant for tighter paper alignment
- Measure token-wise depth-attention entropy over training
- Add representation-similarity analysis between early and late layers
- Run structured early-layer dropout or freezing experiments
- Compare against DenseFormer-style static cross-layer mixing
- Study how the preferred depth/width ratio shifts under matched compute budgets
