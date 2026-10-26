# qwenimage21_layout.sh — sourced (not run) by the *_qwenimage21.sh experiment scripts.
#
# Qwen/Qwen-Image-2.1 is ~32 GiB of bf16 weights (Qwen3-VL text encoder 17 GB, transformer 14 GB,
# VAE 1.3 GB), ~38 GiB with render activations at 1024x1024 (IMAGE_GIB, an estimate: check it with
# a smoke test). It is not step-distilled: the pipeline default is 40 steps, and its classifier-free
# guidance (true_cfg_scale) only runs with a negative prompt, which the loop driver never passes,
# so guidance 1.0 = no CFG, as in the pipeline default.
#
# Sets the GPU layout from GPU_GROUPS and the size of its cards (each value can still be set
# explicitly):
#   VLLM_TENSOR_PARALLEL_SIZE  VLM_TP (set by the script: 1, or "group" = all GPUs of a group)
#   IMAGE_DEVICE               the group's last GPU
#   VLLM_GPU_MEMORY_UTILIZATION  VLM_GIB split over the tensor-parallel GPUs, or 0.8 when the VLM
#                              has its cards to itself (Qwen-Image on a GPU of its own)
#   MIN_FREE_MB                what the busiest GPU of a group needs
# Expects VLM_GIB, VLM_TP and GPU_GROUPS from the script.

export IMAGE_MODEL="Qwen/Qwen-Image-2.1"
export RENDER_STEPS="${RENDER_STEPS:-40}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
IMAGE_GIB="${IMAGE_GIB:-38}"

# Layout from the first group (all groups are assumed to look alike) and the size of its first card.
first_group="${GPU_GROUPS%% *}"
IFS=',' read -ra first_gpus <<< "$first_group"
n_gpus="${#first_gpus[@]}"
card_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${first_gpus[0]}" 2>/dev/null || true)"
card_mib="${card_mib:-46068}"  # no nvidia-smi: assume an A40
if [[ "$VLM_TP" == "group" ]]; then VLM_TP="$n_gpus"; fi
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$VLM_TP}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:$((n_gpus - 1))}"
image_idx="${IMAGE_DEVICE#cuda}"; image_idx="${image_idx#:}"; image_idx="${image_idx:-0}"
# Does Qwen-Image share a card with the VLM?
shared=0; if (( image_idx < VLLM_TENSOR_PARALLEL_SIZE )); then shared=1; fi
if (( shared )); then
  # Share of each card for vLLM, rounded up to 0.01.
  default_util="$(awk -v v="$VLM_GIB" -v n="$VLLM_TENSOR_PARALLEL_SIZE" -v c="$card_mib" \
    'BEGIN { u = v * 1024 / n / c; u = int(u * 100 + 0.999) / 100; printf "%.2f", u }')"
else
  default_util="0.8"
fi
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$default_util}"
export MIN_FREE_MB="${MIN_FREE_MB:-$(awk -v u="$VLLM_GPU_MEMORY_UTILIZATION" -v c="$card_mib" -v i="$IMAGE_GIB" -v s="$shared" \
  'BEGIN { v = u * c; m = i * 1024; printf "%d", s ? v + m : (v > m ? v : m) }')}"
if (( shared )) && awk -v u="$VLLM_GPU_MEMORY_UTILIZATION" -v c="$card_mib" -v i="$IMAGE_GIB" 'BEGIN { exit !(u * c + i * 1024 > c) }'; then
  echo "WARNING: vLLM share ${VLLM_GPU_MEMORY_UTILIZATION} of a ${card_mib} MiB card leaves less than ${IMAGE_GIB} GiB for" \
       "Qwen-Image-2.1; use more GPUs per group (e.g. GPU_GROUPS=\"0,1 2,3\")." >&2
fi
echo "layout: ${n_gpus} GPU(s)/group of ${card_mib} MiB, tensor parallel ${VLLM_TENSOR_PARALLEL_SIZE}," \
     "vLLM share ${VLLM_GPU_MEMORY_UTILIZATION}, Qwen-Image-2.1 on ${IMAGE_DEVICE}, min free ${MIN_FREE_MB} MiB"
