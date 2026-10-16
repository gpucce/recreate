#!/usr/bin/env bash
#
# VLStereoSet, both models swapped: google/gemma-4-E4B-it describes, black-forest-labs/FLUX.2-klein-4B
# renders (4 steps, no classifier-free guidance; see vlstereoset_qwen3vl4b_flux2klein.sh).
# Compare with the baseline data_bkp/vlstereoset_runs/ and with vlstereoset_qwen3vl4b_flux2klein.sh.
#
# Gemma 4 E4B is ~16 GB of bf16 weights (twice Qwen3-VL-4B), so vLLM gets 50% of the card
# (~23 GB: weights + KV cache) next to klein (~15 GB), batch 4, one process per GPU. Thinking is
# turned off (the driver passes enable_thinking=False). On an out-of-memory error lower BATCH_SIZE.
#
#   experiment_scripts/vlstereoset_gemma4e4b_flux2klein.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_gemma4e4b_flux2klein.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/vlstereoset_gemma4e4b_flux2klein.sh   # only GPU 2
#   experiment_scripts/vlstereoset_gemma4e4b_flux2klein.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="gemma4e4b_flux2klein"
export VISION_MODEL="google/gemma-4-E4B-it"
export IMAGE_MODEL="black-forest-labs/FLUX.2-klein-4B"
export VLM_DTYPE="${VLM_DTYPE:-bfloat16}"              # released in bf16; float16 risks overflow
export RENDER_STEPS="${RENDER_STEPS:-4}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
fi
export MIN_FREE_MB="${MIN_FREE_MB:-44000}"             # vLLM ~23 GB + klein ~15 GB + render activations
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_launch.sh" "$@"
