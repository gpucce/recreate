#!/usr/bin/env bash
#
# experiment_launch.sh — shared background launcher for the model experiments (VLStereoSet by
# default; DATASET=coco_dog or coco_car runs the same experiment on a COCO subset, see the
# coco_*.sh wrappers).
#
# Not meant to be called directly: each experiment script in this folder sets EXPERIMENT,
# VISION_MODEL, IMAGE_MODEL and its GPU layout, then execs this file. It runs
# content_convergence/run_loop_optimized.py (models loaded once per process) with one detached
# process per GPU GROUP. A group is one GPU ("2") or several ("0,1") for models that need two cards.
#
# Every experiment reuses the baseline's selection (seed 0, the whole vlstereoset_clean split, 16
# loop steps, 1024x1024), so its image_<row>/ dirs line up row for row with the baseline in
# data_bkp/vlstereoset_runs/ (Qwen3-VL-4B-Instruct + Z-Image-Turbo), which this launcher never
# touches. The VLM runs on vLLM by default (~5x faster than the transformers pipeline the
# baseline used, at batch 4; same sampling settings); VLM_BACKEND=transformers matches the
# baseline's backend instead. vLLM needs the conda env ./vllm_venv (CUDA 13.0 toolkit), which
# this launcher activates. Output goes to
#
#     $OUT_ROOT/<run name>/{config.json, manifest/, logs/, image_<row>/...}
#     OUT_ROOT=data_bkp/<dataset>_experiments/<experiment>,  run name = launch time YYYYmmdd-HHMMSS
#
# and $OUT_ROOT/latest points at the last launched run.
#
# Subcommands (pass them to the experiment script):
#   --dry-run    print the per-group plan and commands; launch nothing, write nothing
#   --status     which processes of the latest run (or RUN_NAME) are alive
#   --stop       SIGTERM (then SIGKILL) them
#
# Env vars (all optional; the experiment script sets the model ones):
#   DATASET=vlstereoset    vlstereoset | coco_dog | coco_car (any --dataset of run_bbq_loop.py)
#   NUM_IMAGES=all STEPS=16 SEED=0 WIDTH=1024 HEIGHT=1024 MAX_CONSECUTIVE_FAILURES=3
#   VLM_BACKEND=vllm       vllm | transformers
#   BATCH_SIZE=4           chains advanced together (default 4 with vllm, 1 with transformers)
#   GPU_GROUPS="0 1 2 3"   space-separated groups, e.g. "0,1 2,3" or "0,1,2,3"
#   MIN_FREE_MB=35000      free VRAM every GPU of a group needs; busy groups are skipped
#   FORCE_GPU=0            1 = skip the free-VRAM check
#   VLLM_TENSOR_PARALLEL_SIZE=1        vllm: split the VLM evenly over this many GPUs of a group
#   VLLM_GPU_MEMORY_UTILIZATION=""     vllm: share of each GPU for the VLM (blank = driver default 0.4)
#   VLM_MAX_MEMORY=""      transformers: split the VLM over a group's GPUs, e.g. "0:38GiB,1:20GiB"
#   IMAGE_DEVICE=cuda      diffusion device within the group, e.g. cuda:1
#   VLM_DTYPE=""           float16 | bfloat16 (blank = describe.py default, float16)
#   RENDER_STEPS="" GUIDANCE_SCALE=""   blank = create.py defaults (16 steps, guidance 0.0)
#   RUN_NAME=""            blank = new run named after the launch time; an existing name finishes
#                          that run (finished rows are skipped, unfinished ones redone from image_0)
#   DATA_DIR=data_bkp OUT_ROOT=$DATA_DIR/${DATASET}_experiments/$EXPERIMENT
#   VLLM_ENV=vllm_venv     conda env activated for vllm (conda_venv if vllm_venv is missing); PY=conda_venv/bin/python for transformers

set -euo pipefail

: "${EXPERIMENT:?set by the experiment script}"
: "${VISION_MODEL:?set by the experiment script}"
: "${IMAGE_MODEL:?set by the experiment script}"
DATASET="${DATASET:-vlstereoset}"
NUM_IMAGES="${NUM_IMAGES:-all}"
STEPS="${STEPS:-16}"
SEED="${SEED:-0}"
VLM_BACKEND="${VLM_BACKEND:-vllm}"
case "$VLM_BACKEND" in
  vllm)         BATCH_SIZE="${BATCH_SIZE:-4}" ;;
  transformers) BATCH_SIZE="${BATCH_SIZE:-1}" ;;
  *) echo "ERROR: unknown VLM_BACKEND=$VLM_BACKEND (expected vllm or transformers)" >&2; exit 1 ;;
esac
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"
GPU_GROUPS="${GPU_GROUPS:-0 1 2 3}"
MIN_FREE_MB="${MIN_FREE_MB:-35000}"
FORCE_GPU="${FORCE_GPU:-0}"
VLM_MAX_MEMORY="${VLM_MAX_MEMORY:-}"
IMAGE_DEVICE="${IMAGE_DEVICE:-cuda}"
VLM_DTYPE="${VLM_DTYPE:-}"
RENDER_STEPS="${RENDER_STEPS:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
DATA_DIR="${DATA_DIR:-data_bkp}"
OUT_ROOT="${OUT_ROOT:-$DATA_DIR/${DATASET}_experiments/$EXPERIMENT}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # experiment_scripts/
ROOT="$(dirname "$HERE")"                              # repo root
if [[ "$VLM_BACKEND" == "vllm" && "${1:-}" != "--status" && "${1:-}" != "--stop" ]]; then
  # vLLM needs its conda env activated (CUDA toolkit and build tools on PATH), not only its python.
  # Default ./vllm_venv; machines without it use ./conda_venv, which also has vllm.
  if [[ -z "${VLLM_ENV:-}" ]]; then
    if [[ -d "$ROOT/vllm_venv" ]]; then VLLM_ENV="$ROOT/vllm_venv"; else VLLM_ENV="$ROOT/conda_venv"; fi
  fi
  set +u   # conda's activation scripts read unset variables
  eval "$(conda shell.bash hook)"
  conda activate "$VLLM_ENV"
  set -u
  PY="${PY:-$VLLM_ENV/bin/python}"
fi
PY="${PY:-$ROOT/conda_venv/bin/python}"
# CUDA 13 forward-compat libs (conda package cuda-compat), for drivers older than the env's CUDA.
# conda activation adds them too; this covers the transformers path, which calls $PY directly.
COMPAT_DIR="$(dirname "$(dirname "$PY")")/cuda-compat"
if [[ -d "$COMPAT_DIR" && ":${LD_LIBRARY_PATH:-}:" != *":$COMPAT_DIR:"* ]]; then
  export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
DRIVER="$ROOT/content_convergence/run_loop_optimized.py"
SELF="experiment_scripts/${DATASET}_${EXPERIMENT}.sh"   # the experiment script, for the hints below

# Relative paths are taken from the repo root, so the scripts can be launched from anywhere.
case "$DATA_DIR" in /*) : ;; *) DATA_DIR="$ROOT/$DATA_DIR" ;; esac
case "$OUT_ROOT" in /*) : ;; *) OUT_ROOT="$ROOT/$OUT_ROOT" ;; esac

case "${1:-}" in
  --status|--stop) RUN_NAME="${RUN_NAME:-latest}" ;;
  *)               RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}" ;;
esac
RUN_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
PID_FILE="$LOG_DIR/launch.pids"

gpu_free_mb() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr ',' ' ' | awk -v i="$1" '$1==i { print $2-$3; exit }'
}

# ---- subcommands ---------------------------------------------------------
if [[ "${1:-}" == "--status" || "${1:-}" == "--stop" ]]; then
  if [[ ! -f "$PID_FILE" ]]; then echo "no pid file at $PID_FILE (nothing tracked)." >&2; exit 1; fi
fi
if [[ "${1:-}" == "--status" ]]; then
  echo "$EXPERIMENT, run $(basename "$(readlink -f "$RUN_DIR")") ($PID_FILE):"
  while read -r pid tag; do
    [[ -z "${pid:-}" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then echo "  RUNNING   pid=$pid  $tag"; else echo "  FINISHED  pid=$pid  $tag"; fi
  done < "$PID_FILE"
  exit 0
fi
if [[ "${1:-}" == "--stop" ]]; then
  for pass in TERM KILL; do
    while read -r pid tag; do
      [[ -z "${pid:-}" ]] && continue
      if kill -0 "$pid" 2>/dev/null; then
        kill -s "$pass" "$pid" 2>/dev/null && echo "  sent SIG$pass -> $pid ($tag)"
        if [[ "$pass" == "TERM" ]]; then sleep 2; fi
      fi
    done < "$PID_FILE"
    sleep 2
  done
  exit 0
fi
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ---- pre-flight: keep the groups whose every GPU has enough free VRAM -----
read -ra GROUPS_REQUESTED <<< "$GPU_GROUPS"
EFF_GROUPS=()
echo "$DATASET experiment: $EXPERIMENT"
echo "  models        : $VISION_MODEL + $IMAGE_MODEL"
echo "  GPU groups    : $GPU_GROUPS   (min free ${MIN_FREE_MB} MiB per GPU, force=${FORCE_GPU})"
for group in "${GROUPS_REQUESTED[@]}"; do
  ok=1
  for g in ${group//,/ }; do
    [[ "$FORCE_GPU" == "1" ]] && continue
    free="$(gpu_free_mb "$g")"
    if [[ -z "$free" ]]; then
      echo "  gpu $g : no VRAM info -> using anyway (verify manually!)"
    elif (( free < MIN_FREE_MB )); then
      echo "  gpu $g : only ${free} MiB free (< ${MIN_FREE_MB}) -> skipping group $group"; ok=0
    else
      echo "  gpu $g : ${free} MiB free -> OK"
    fi
  done
  if (( ok )); then EFF_GROUPS+=("$group"); fi
done
N_GROUPS="${#EFF_GROUPS[@]}"
if (( N_GROUPS == 0 )); then
  echo "ERROR: no usable GPU group. Free some VRAM, or set FORCE_GPU=1 / MIN_FREE_MB lower." >&2
  exit 1
fi

if [[ "$NUM_IMAGES" == "all" ]]; then
  NUM_IMAGES="$("$PY" "$DRIVER" --dataset "$DATASET" --data-dir "$DATA_DIR" --count-rows)"
fi
echo "  usable groups : ${EFF_GROUPS[*]}  (${N_GROUPS} process(es), group i takes positions i, i+${N_GROUPS}, ...)"
echo "  images / steps: $NUM_IMAGES rows (seed $SEED), $STEPS loop steps, batch size $BATCH_SIZE"
echo "  VLM backend   : $VLM_BACKEND$([[ "$VLM_BACKEND" == vllm ]] && echo " (env $VLLM_ENV, tensor parallel $VLLM_TENSOR_PARALLEL_SIZE)")"
echo "  render        : steps=${RENDER_STEPS:-create.py default}  guidance=${GUIDANCE_SCALE:-create.py default}"
echo "  run dir       : $RUN_DIR"
if (( ! DRY_RUN )); then
  mkdir -p "$LOG_DIR"
  : > "$PID_FILE"
fi

# ---- one detached driver per usable group ----------------------------------
for i in "${!EFF_GROUPS[@]}"; do
  group="${EFF_GROUPS[$i]}"
  log="$LOG_DIR/shard${i}_gpu${group//,/-}.log"
  cmd=("$PY" "$DRIVER" --dataset "$DATASET" --data-dir "$DATA_DIR"
       --start-index 0 --num-images "$NUM_IMAGES" --shard "$i" --num-shards "$N_GROUPS"
       --seed "$SEED" --steps "$STEPS" --batch-size "$BATCH_SIZE"
       --max-consecutive-failures "$MAX_CONSECUTIVE_FAILURES"
       --gpus "$group" --image-device "$IMAGE_DEVICE"
       --vision-model "$VISION_MODEL" --image-model "$IMAGE_MODEL"
       --width "$WIDTH" --height "$HEIGHT"
       --out "$OUT_ROOT" --run-name "$RUN_NAME")
  cmd+=(--vlm-backend "$VLM_BACKEND")
  if [[ "$VLM_BACKEND" == "vllm" ]]; then
    cmd+=(--vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
    if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
      cmd+=(--vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
    fi
  fi
  if [[ -n "$VLM_MAX_MEMORY" ]]; then cmd+=(--vlm-max-memory "$VLM_MAX_MEMORY"); fi
  if [[ -n "$VLM_DTYPE" ]]; then cmd+=(--vlm-dtype "$VLM_DTYPE"); fi
  if [[ -n "$RENDER_STEPS" ]]; then cmd+=(--render-steps "$RENDER_STEPS"); fi
  if [[ -n "$GUIDANCE_SCALE" ]]; then cmd+=(--guidance-scale "$GUIDANCE_SCALE"); fi

  if (( DRY_RUN )); then
    echo "  [dry-run] shard $i/$N_GROUPS on gpus $group  (log: $log)"
    printf '          would run:'
    for _x in "${cmd[@]}"; do printf ' %q' "$_x"; done
    echo
    continue
  fi
  nohup "${cmd[@]}" >> "$log" 2>&1 &
  pid=$!
  echo "$pid shard=$i/$N_GROUPS gpus=$group" >> "$PID_FILE"
  echo "  LAUNCHED shard $i/$N_GROUPS on gpus $group  pid=$pid  (log: $log)"
done

if (( DRY_RUN )); then
  echo
  echo "(dry-run: nothing launched, no files written. Check a shard's selection and config.json with:"
  echo "   ${cmd[*]} --dry-run )"
  exit 0
fi
ln -sfn "$RUN_NAME" "$OUT_ROOT/latest"
echo
echo "monitor : tail -f $LOG_DIR/shard0_*.log    $SELF --status"
echo "stop    : $SELF --stop"
echo "resume  : RUN_NAME=$RUN_NAME $SELF"
