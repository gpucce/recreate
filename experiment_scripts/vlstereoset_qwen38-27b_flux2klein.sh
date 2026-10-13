#!/usr/bin/env bash
#
# VLStereoSet, both models swapped: Qwen/Qwen3.8-27B describes, black-forest-labs/FLUX.2-klein-4B
# renders (4 steps, no classifier-free guidance; see vlstereoset_qwen3vl4b_flux2klein.sh).
# Compare with the baseline data_bkp/vlstereoset_runs/ and with the two single-swap experiments.
#
# Same layout as vlstereoset_qwen38-27b_zimage.sh: one process over all four A40s, the VLM on
# vLLM tensor-parallel over the four (~13 GB of weights each, 40% of each card), klein on the
# fourth, batch 8, thinking off. Not run yet: try a smoke test first.
#
# Fallback with the transformers backend, two processes of two GPUs each:
#   VLM_BACKEND=transformers GPU_GROUPS="0,1 2,3" VLM_MAX_MEMORY="0:38GiB,1:20GiB" \
#     IMAGE_DEVICE=cuda:1 experiment_scripts/vlstereoset_qwen38-27b_flux2klein.sh
#
#   experiment_scripts/vlstereoset_qwen38-27b_flux2klein.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_qwen38-27b_flux2klein.sh              # launch in the background
#   experiment_scripts/vlstereoset_qwen38-27b_flux2klein.sh --status | --stop
# Other env vars: see vlstereoset_launch.sh.

set -euo pipefail
export EXPERIMENT="qwen38-27b_flux2klein"
export VISION_MODEL="Qwen/Qwen3.8-27B"
export IMAGE_MODEL="black-forest-labs/FLUX.2-klein-4B"
export VLM_DTYPE="${VLM_DTYPE:-bfloat16}"              # released in bf16; float16 risks overflow
export RENDER_STEPS="${RENDER_STEPS:-4}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export GPU_GROUPS="${GPU_GROUPS:-0,1,2,3}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then export BATCH_SIZE="${BATCH_SIZE:-8}"; fi   # one process: batch more
export MIN_FREE_MB="${MIN_FREE_MB:-44000}"
export LAUNCHED_AS="$0"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_launch.sh" "$@"
