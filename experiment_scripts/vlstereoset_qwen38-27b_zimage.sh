#!/usr/bin/env bash
#
# VLStereoSet, text-model swap: Qwen/Qwen3.8-27B describes, Tongyi-MAI/Z-Image-Turbo renders
# (baseline render settings: create.py defaults, 16 steps, guidance 0.0).
# Compare with the baseline data_bkp/vlstereoset_runs/ (Qwen3-VL-4B-Instruct + Z-Image-Turbo).
#
# Qwen3.8-27B is ~52 GB in bf16, more than one A40 holds. vLLM splits a model evenly, and on two
# GPUs (26 GB each plus overhead) no card would have room left for Z-Image (~21 GB), so this runs
# ONE process over all four A40s: the VLM tensor-parallel over the four (~13 GB of weights each,
# 40% of each card), Z-Image on the fourth, batch 8. Thinking is turned off (the model's chat
# template enables it by default), so only the description reaches the prompt files.
# Not run yet: try a smoke test first (NUM_IMAGES=2 STEPS=2 OUT_ROOT=data_bkp/smoke ...).
#
# Fallback with the transformers backend, two processes of two GPUs each:
#   VLM_BACKEND=transformers GPU_GROUPS="0,1 2,3" VLM_MAX_MEMORY="0:38GiB,1:20GiB" \
#     IMAGE_DEVICE=cuda:1 experiment_scripts/vlstereoset_qwen38-27b_zimage.sh
#
#   experiment_scripts/vlstereoset_qwen38-27b_zimage.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_qwen38-27b_zimage.sh              # launch in the background
#   experiment_scripts/vlstereoset_qwen38-27b_zimage.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="qwen38-27b_zimage"
export VISION_MODEL="Qwen/Qwen3.8-27B"
export IMAGE_MODEL="Tongyi-MAI/Z-Image-Turbo"
export VLM_DTYPE="${VLM_DTYPE:-bfloat16}"              # released in bf16; float16 risks overflow
export GPU_GROUPS="${GPU_GROUPS:-0,1,2,3}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then export BATCH_SIZE="${BATCH_SIZE:-8}"; fi   # one process: batch more
export MIN_FREE_MB="${MIN_FREE_MB:-44000}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_launch.sh" "$@"
