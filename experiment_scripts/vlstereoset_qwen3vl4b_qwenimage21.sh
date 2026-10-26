#!/usr/bin/env bash
#
# VLStereoSet, image-model swap: Qwen/Qwen3-VL-4B-Instruct describes (as in the baseline),
# Qwen/Qwen-Image-2.1 renders (40 steps, no classifier-free guidance; see qwenimage21_layout.sh).
# Compare with the baseline data_bkp/vlstereoset_runs/ (Qwen3-VL-4B-Instruct + Z-Image-Turbo).
#
# Qwen-Image-2.1 (~38 GiB with activations) is about twice Z-Image, so the layout follows the cards:
#   H100 (80 GB): GPU_GROUPS="3 4"          one GPU per process, vLLM ~14 GiB next to Qwen-Image
#   4x A40 (48 GB): GPU_GROUPS="0,1 2,3"    the VLM alone on the first card, Qwen-Image on the second
# Not run yet: try a smoke test first (NUM_IMAGES=2 STEPS=2 OUT_ROOT=data_bkp/smoke GPU_GROUPS="3" ...);
# on an out-of-memory error lower BATCH_SIZE to 2 (the renders are batched too).
#
#   experiment_scripts/vlstereoset_qwen3vl4b_qwenimage21.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_qwen3vl4b_qwenimage21.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/vlstereoset_qwen3vl4b_qwenimage21.sh   # only GPU 2
#   experiment_scripts/vlstereoset_qwen3vl4b_qwenimage21.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="qwen3vl4b_qwenimage21"
export VISION_MODEL="Qwen/Qwen3-VL-4B-Instruct"
export GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
VLM_GIB="${VLM_GIB:-14}"       # 9 GB of weights + KV cache (the 30% of an A40 the zimage script gives it)
VLM_TP=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/qwenimage21_layout.sh"
exec "$HERE/experiment_launch.sh" "$@"
