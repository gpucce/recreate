#!/usr/bin/env bash
#
# run_vlstereoset_clean_200.sh — background launcher for the VLStereoSet content-convergence run.
#
# Same as run_bbq_200.sh but on vlstereoset_clean
# (datasets.load_from_disk("data/vlstereoset_clean"), 1273 rows), writing under
# data/vlstereoset_runs/ by default. All env vars and subcommands of run_bbq_200.sh
# apply unchanged; DATASET is forced to vlstereoset.
#
# Usage (from the repo root, or anywhere):
#   content_convergence/run_vlstereoset_clean_200.sh                          # 200 random images, 16 steps
#   GPUS="0 2 3" content_convergence/run_vlstereoset_clean_200.sh             # choose GPUs
#   NUM_IMAGES=50 STEPS=4 content_convergence/run_vlstereoset_clean_200.sh    # quick smoke test
#   content_convergence/run_vlstereoset_clean_200.sh --status | --stop | --dry-run
#
# Monitor:
#   tail -f data/vlstereoset_runs/logs/gpu0.log

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=vlstereoset OUT_DIR="${OUT_DIR:-data/vlstereoset_runs}" exec "$HERE/run_bbq_200.sh" "$@"
