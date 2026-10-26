#!/usr/bin/env bash
#
# VLStereoSet, both models swapped: Qwen/Qwen3.8-27B describes, Qwen/Qwen-Image-2.1 renders
# (40 steps, no classifier-free guidance; see qwenimage21_layout.sh).
# Compare with the baseline data_bkp/vlstereoset_runs/ and with the two single-swap experiments.
#
# One process per GPU group, each running the VLM on vLLM tensor-parallel over its group's GPUs and
# Qwen-Image-2.1 on the group's last GPU, batch 8, thinking off. VLM ~56 GiB + Qwen-Image ~38 GiB
# do not fit one H100, so the default is two-GPU groups:
#   2x H100 (80 GB): GPU_GROUPS="4,5"        (~28 GiB of VLM per card, Qwen-Image on the second)
#   4x H100        : GPU_GROUPS="4,5 6,7"    (two independent processes)
# On 4x A40 (48 GB) it does not fit tensor-parallel (14 GiB + 38 GiB > 46 GiB): use three cards for
# the VLM and give Qwen-Image the fourth, if the model's head count allows tensor parallel 3,
#   GPU_GROUPS="0,1,2,3" VLLM_TENSOR_PARALLEL_SIZE=3 experiment_scripts/vlstereoset_qwen38-27b_qwenimage21.sh
# Not run yet: try a smoke test first (NUM_IMAGES=2 STEPS=2 OUT_ROOT=data_bkp/smoke ...).
#
#   experiment_scripts/vlstereoset_qwen38-27b_qwenimage21.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_qwen38-27b_qwenimage21.sh              # launch in the background
#   experiment_scripts/vlstereoset_qwen38-27b_qwenimage21.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="qwen38-27b_qwenimage21"
export VISION_MODEL="Qwen/Qwen3.8-27B"
export VLM_DTYPE="${VLM_DTYPE:-bfloat16}"              # released in bf16; float16 risks overflow
export GPU_GROUPS="${GPU_GROUPS:-0,1 2,3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then export BATCH_SIZE="${BATCH_SIZE:-8}"; fi   # big VLM: batch more
VLM_GIB="${VLM_GIB:-56}"       # VLM weights (51.1 GiB in bf16) + ~3 GiB profiling peak + ~2 GiB KV cache, whole group
VLM_TP=group
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/qwenimage21_layout.sh"
exec "$HERE/experiment_launch.sh" "$@"
