#!/usr/bin/env bash
#
# VLStereoSet baseline models: Qwen/Qwen3-VL-4B-Instruct describes, Tongyi-MAI/Z-Image-Turbo renders
# (create.py defaults: 16 steps, guidance 0.0) -- the same models as data_bkp/vlstereoset_runs/.
#
# That existing baseline was made by run_bbq_loop.py with the transformers VLM pipeline, batch 1,
# and stays as it is. This script reruns the baseline models through the same driver and backend
# as the other experiments here (vLLM, batch 4, thinking off), so a model swap can be compared
# with no backend difference. VLM_BACKEND=transformers BATCH_SIZE=1 reproduces the original
# backend instead (~40 s/step).
#
# Z-Image (~21 GB with its text encoder) is larger than klein, so vLLM gets 30% of the card
# (~14 GB: 9 GB of weights + KV cache) instead of the default 40%. Not run yet with vLLM: try a
# smoke test first (NUM_IMAGES=2 STEPS=2 OUT_ROOT=data_bkp/smoke GPU_GROUPS="2" ...); on an out-of-memory
# error lower BATCH_SIZE to 2.
#
#   experiment_scripts/vlstereoset_qwen3vl4b_zimage.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_qwen3vl4b_zimage.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/vlstereoset_qwen3vl4b_zimage.sh   # only GPU 2
#   experiment_scripts/vlstereoset_qwen3vl4b_zimage.sh --status | --stop
# Other env vars: see experiment_launch.sh.

set -euo pipefail
export EXPERIMENT="qwen3vl4b_zimage"
export VISION_MODEL="Qwen/Qwen3-VL-4B-Instruct"
export IMAGE_MODEL="Tongyi-MAI/Z-Image-Turbo"
export GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"   # leave room for Z-Image
fi
export MIN_FREE_MB="${MIN_FREE_MB:-44000}"             # vLLM ~14 GB + Z-Image ~21 GB + render activations
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_launch.sh" "$@"
