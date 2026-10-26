#!/usr/bin/env bash
#
# VLStereoSet, both models swapped: google/gemma-4-E4B-it describes, Qwen/Qwen-Image-2.1 renders
# (40 steps, no classifier-free guidance; see qwenimage21_layout.sh).
# Compare with the baseline data_bkp/vlstereoset_runs/ and with vlstereoset_qwen3vl4b_qwenimage21.sh.
#
# Gemma 4 E4B is ~16 GB of bf16 weights, vLLM gets ~21 GiB (weights + KV cache), Qwen-Image-2.1
# ~38 GiB, batch 2. Thinking is turned off (the driver passes enable_thinking=False). Layouts:
#   H100 (80 GB): GPU_GROUPS="3 4"          one GPU per process
#   4x A40 (48 GB): GPU_GROUPS="0,1 2,3"    the VLM alone on the first card, Qwen-Image on the second
# Not run yet: try a smoke test first (NUM_IMAGES=2 STEPS=2 OUT_ROOT=data_bkp/smoke GPU_GROUPS="3" ...);
# on an out-of-memory error lower BATCH_SIZE to 1.
#
#   experiment_scripts/vlstereoset_gemma4e4b_qwenimage21.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_gemma4e4b_qwenimage21.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/vlstereoset_gemma4e4b_qwenimage21.sh   # only GPU 2
#   experiment_scripts/vlstereoset_gemma4e4b_qwenimage21.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="gemma4e4b_qwenimage21"
export VISION_MODEL="google/gemma-4-E4B-it"
export VLM_DTYPE="${VLM_DTYPE:-bfloat16}"              # released in bf16; float16 risks overflow
export GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then export BATCH_SIZE="${BATCH_SIZE:-2}"; fi
VLM_GIB="${VLM_GIB:-21}"       # the 45% of an A40 the zimage script gives it
VLM_TP=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/qwenimage21_layout.sh"
exec "$HERE/experiment_launch.sh" "$@"
