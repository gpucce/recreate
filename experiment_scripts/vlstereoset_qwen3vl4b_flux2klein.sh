#!/usr/bin/env bash
#
# VLStereoSet, image-model swap: Qwen/Qwen3-VL-4B-Instruct describes (as in the baseline),
# black-forest-labs/FLUX.2-klein-4B renders.
# Compare with the baseline data_bkp/vlstereoset_runs/ (Qwen3-VL-4B-Instruct + Z-Image-Turbo).
#
# FLUX.2-klein-4B is step-distilled: 4 diffusion steps, and its pipeline never applies
# classifier-free guidance (guidance 1.0 just avoids the pipeline's warning).
# The VLM runs on vLLM (40% of the GPU) next to klein, batch 4, one process per GPU. Measured on
# one A40: 4.3 s per loop step, so the full run (1273 rows x 16 steps) is ~24 h on one GPU,
# ~6 h on four. VLM_BACKEND=transformers BATCH_SIZE=1 uses the baseline's backend (~23 s/step).
#
#   experiment_scripts/vlstereoset_qwen3vl4b_flux2klein.sh --dry-run    # plan only
#   experiment_scripts/vlstereoset_qwen3vl4b_flux2klein.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/vlstereoset_qwen3vl4b_flux2klein.sh   # only GPU 2
#   experiment_scripts/vlstereoset_qwen3vl4b_flux2klein.sh --status | --stop
# Other env vars: see vlstereoset_launch.sh.

set -euo pipefail
export EXPERIMENT="qwen3vl4b_flux2klein"
export VISION_MODEL="Qwen/Qwen3-VL-4B-Instruct"
export IMAGE_MODEL="black-forest-labs/FLUX.2-klein-4B"
export RENDER_STEPS="${RENDER_STEPS:-4}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
export GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
export MIN_FREE_MB="${MIN_FREE_MB:-42000}"             # vLLM ~18 GB + klein ~15 GB + render activations
export LAUNCHED_AS="$0"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_launch.sh" "$@"
