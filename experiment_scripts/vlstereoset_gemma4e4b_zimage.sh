#!/usr/bin/env bash
#
# VLStereoSet, VLM swap: google/gemma-4-E4B-it describes, Tongyi-MAI/Z-Image-Turbo renders
# (baseline render settings: create.py defaults, 16 steps, guidance 0.0).
# Compare with the baseline data_bkp/vlstereoset_runs/ and with vlstereoset_qwen3vl4b_zimage.sh.
#
# Gemma 4 E4B is ~16 GB of bf16 weights (twice Qwen3-VL-4B), so vLLM gets 45% of the card
# (~20.7 GB: weights + a small KV cache), Z-Image (~21 GB) the rest, batch 2, one process per GPU.
# Thinking is turned off (the driver passes enable_thinking=False), so only the description
# reaches the prompt files. On an out-of-memory error lower BATCH_SIZE to 1, or run two-GPU groups
# (VLM alone on one card, Z-Image on the other):
#   GPU_GROUPS="0,1 2,3" VLLM_GPU_MEMORY_UTILIZATION=0.8 IMAGE_DEVICE=cuda:1 BATCH_SIZE=4 \
#     MIN_FREE_MB=40000 experiment_scripts/vlstereoset_gemma4e4b_zimage.sh
#
#   experiment_scripts/vlstereoset_gemma4e4b_zimage.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_gemma4e4b_zimage.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/vlstereoset_gemma4e4b_zimage.sh   # only GPU 2
#   experiment_scripts/vlstereoset_gemma4e4b_zimage.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="gemma4e4b_zimage"
export VISION_MODEL="google/gemma-4-E4B-it"
export IMAGE_MODEL="Tongyi-MAI/Z-Image-Turbo"
export VLM_DTYPE="${VLM_DTYPE:-bfloat16}"              # released in bf16; float16 risks overflow
export GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
  export BATCH_SIZE="${BATCH_SIZE:-2}"
fi
export MIN_FREE_MB="${MIN_FREE_MB:-45000}"             # vLLM ~21 GB + Z-Image ~21 GB + render activations
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_launch.sh" "$@"
