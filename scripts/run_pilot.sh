#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --config configs/pilot_t4.yaml \
  --overrides experiment.name=pilot_baseline model.architecture=baseline model.attnres.enabled=false

python -m src.train \
  --config configs/pilot_t4.yaml \
  --overrides experiment.name=pilot_attnres model.architecture=attnres model.attnres.enabled=true
