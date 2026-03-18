#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/attnres_t4_small.yaml
