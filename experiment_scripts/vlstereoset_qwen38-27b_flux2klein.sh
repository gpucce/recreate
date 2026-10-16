#!/usr/bin/env bash
#
# VLStereoSet, both models swapped: Qwen/Qwen3.8-27B describes, black-forest-labs/FLUX.2-klein-4B
# renders (4 steps, no classifier-free guidance; see vlstereoset_qwen3vl4b_flux2klein.sh).
# Compare with the baseline data_bkp/vlstereoset_runs/ and with the two single-swap experiments.
#
# One process per GPU group, each running the VLM on vLLM tensor-parallel over its group's GPUs and
# klein on the group's last GPU, batch 8, thinking off. The layout is derived from GPU_GROUPS and
# the size of its cards, so the same script runs on e.g.
#   4x A40 (48 GB) : GPU_GROUPS="0,1,2,3"   (default; ~13 GB of VLM weights per card)
#   1x H100 (80 GB): GPU_GROUPS="6"          (VLM ~56 GiB + klein ~17 GiB: needs the card nearly empty)
#   2x H100        : GPU_GROUPS="4 6"        (two independent single-GPU processes)
# Defaults: tensor parallel = GPUs per group, IMAGE_DEVICE = the group's last GPU, vLLM's share of
# each card = just enough for VLM_GIB split over the group (at least 0.4), MIN_FREE_MB = that
# share plus IMAGE_GIB. Every one of them can still be set explicitly.
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
VLM_GIB="${VLM_GIB:-56}"       # VLM weights (51.1 GiB in bf16) + ~3 GiB profiling peak + ~2 GiB KV cache, whole group
IMAGE_GIB="${IMAGE_GIB:-17}"   # klein: transformer + Qwen3-4B text encoder + VAE + activations

# Layout from the first group (all groups are assumed to look alike) and the size of its first card.
first_group="${GPU_GROUPS%% *}"
IFS=',' read -ra first_gpus <<< "$first_group"
n_gpus="${#first_gpus[@]}"
card_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${first_gpus[0]}" 2>/dev/null || true)"
card_mib="${card_mib:-46068}"  # no nvidia-smi: assume an A40
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$n_gpus}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:$((n_gpus - 1))}"
# Share of each card for vLLM, rounded up to 0.01, never below the driver default 0.4.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$(awk -v v="$VLM_GIB" -v n="$VLLM_TENSOR_PARALLEL_SIZE" -v c="$card_mib" \
  'BEGIN { u = v * 1024 / n / c; u = int(u * 100 + 0.999) / 100; if (u < 0.4) u = 0.4; printf "%.2f", u }')}"
export MIN_FREE_MB="${MIN_FREE_MB:-$(awk -v u="$VLLM_GPU_MEMORY_UTILIZATION" -v c="$card_mib" -v i="$IMAGE_GIB" \
  'BEGIN { printf "%d", u * c + i * 1024 }')}"
if awk -v u="$VLLM_GPU_MEMORY_UTILIZATION" -v c="$card_mib" -v i="$IMAGE_GIB" 'BEGIN { exit !(u * c + i * 1024 > c) }'; then
  echo "WARNING: vLLM share ${VLLM_GPU_MEMORY_UTILIZATION} of a ${card_mib} MiB card leaves less than ${IMAGE_GIB} GiB for klein;" \
       "use more GPUs per group." >&2
fi
echo "layout: ${n_gpus} GPU(s)/group of ${card_mib} MiB, tensor parallel ${VLLM_TENSOR_PARALLEL_SIZE}," \
     "vLLM share ${VLLM_GPU_MEMORY_UTILIZATION}, klein on ${IMAGE_DEVICE}, min free ${MIN_FREE_MB} MiB"
if [[ "${VLM_BACKEND:-vllm}" == "vllm" ]]; then export BATCH_SIZE="${BATCH_SIZE:-8}"; fi   # big VLM: batch more
export LAUNCHED_AS="$0"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_launch.sh" "$@"
