#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/debug_tiny.yaml
