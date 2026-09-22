#!/usr/bin/env bash
#
# run_bbq_200.sh — background launcher for the BBQ-V content-convergence run.
#
# Splits a random sample of NUM_IMAGES BBQ-V images across the (free) GPUs and runs one
# Sequential driver (run_bbq_loop.py) PER GPU, each pinned to its GPU. This
# keeps only ONE {VLM + diffusion} model-set resident per A40 (the VRAM
# constraint from the 5-image test). Every job is detached (nohup) so it
# survives the shell exiting — this is the "run it in the background" entry.
#
# Pre-flight: checks each selected GPU's FREE memory and SKIPS any card that
# can't hold the model-set (prevents OOM on a busy GPU, e.g. one another job
# is using). Override with FORCE_GPU=1 to place jobs on a GPU anyway.
#
# Usage (from the repo root, or anywhere):
#   content_convergence/run_bbq_200.sh                          # defaults below
#   GPUS="0 2 3" content_convergence/run_bbq_200.sh             # choose GPUs (space-separated)
#   NUM_IMAGES=50 STEPS=4 content_convergence/run_bbq_200.sh    # quick smoke test
#   FRESH=0 SEED=1337 content_convergence/run_bbq_200.sh        # keep existing runs; different random sample
#
# Subcommands:
#   content_convergence/run_bbq_200.sh --status                 # which tracked jobs are alive
#   content_convergence/run_bbq_200.sh --stop                   # SIGTERM (then SIGKILL) all jobs this script launched
#   content_convergence/run_bbq_200.sh --dry-run                # show the exact per-GPU plan; launch nothing, write nothing
#
# Monitor:
#   tail -f bbq_runs/logs/gpu0.log            # per-GPU progress
#   nvidia-smi                                # GPU load
#   ls  bbq_runs/image_*/*                    # artifacts piling up (image_N.png / prompt_N.txt)
#
# Env vars (all optional, shown with defaults):
#   NUM_IMAGES=200 STEPS=16 GPUS="0 1 2 3" VISION_MODEL=Qwen/Qwen3-VL-4B-Instruct \
#   IMAGE_MODEL=Tongyi-MAI/Z-Image-Turbo WIDTH=1024 HEIGHT=1024 SEED="" \
#   FRESH=1 MAX_CONSECUTIVE_FAILURES=3 MIN_FREE_MB=35000 FORCE_GPU=0 OUT_DIR=bbq_runs

set -euo pipefail

# ---- config (override via env) -------------------------------------------
NUM_IMAGES="${NUM_IMAGES:-200}"
STEPS="${STEPS:-16}"
GPUS="${GPUS:-0 1 2 3}"
VISION_MODEL="${VISION_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
IMAGE_MODEL="${IMAGE_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
SEED="${SEED:-}"                     # row-sampling seed; blank = run_bbq_loop.py default (0).
                                     # All shards share it, which is what keeps them disjoint.
FRESH="${FRESH:-1}"                  # 1 = wipe+reseed each workdir; 0 = keep, only seed missing
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-35000}"  # min free VRAM a GPU needs to host one process
FORCE_GPU="${FORCE_GPU:-0}"          # 1 = skip the free-VRAM pre-flight
OUT_DIR="${OUT_DIR:-bbq_runs}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # content_convergence/
ROOT="$(dirname "$HERE")"                              # repo root: conda_venv/, bbq_runs/, ...
PY="${PY:-$ROOT/conda_venv/bin/python}"

# A relative OUT_DIR is taken from the repo root, not the caller's cwd,
# so the script can be launched from anywhere (same rule as run_validate_200.sh).
case "$OUT_DIR" in /*) : ;; *) OUT_DIR="$ROOT/$OUT_DIR" ;; esac
LOG_DIR="$OUT_DIR/logs"
MASTER_LOG="$LOG_DIR/bbq_200_master.log"
PID_FILE="$LOG_DIR/bbq_200.pids"

# ---- helpers -------------------------------------------------------------
# free MiB for a GPU index = total - used (empty if index absent / nvidia-smi missing)
gpu_free_mb() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr ',' ' ' \
    | awk -v i="$1" '$1==i { print $2-$3; exit }'
}

# ---- subcommands ---------------------------------------------------------
if [[ ! -f "$PID_FILE" && ("${1:-}" == "--status" || "${1:-}" == "--stop") ]]; then
  echo "no pid file at $PID_FILE (nothing tracked to act on)." >&2
  exit 1
fi

if [[ "${1:-}" == "--status" ]]; then
  echo "tracked jobs ($PID_FILE):"
  while read -r pid tag; do
    [[ -z "${pid:-}" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then echo "  RUNNING   pid=$pid  $tag"; else echo "  FINISHED  pid=$pid  $tag"; fi
  done < "$PID_FILE"
  exit 0
fi

if [[ "${1:-}" == "--stop" ]]; then
  echo "stopping BBQ-V run (pid file: $PID_FILE)"
  # first pass: SIGTERM; second pass: SIGKILL stragglers
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
  echo "done. (child loop.py/python processes under each driver are killed with it.)"
  exit 0
fi

# ---- dry-run: print the plan, launch nothing --------------------------------
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ---- pre-flight: keep only GPUs with enough free VRAM ---------------------
read -ra GPU_REQUESTED <<< "$GPUS"   # word-split on IFS (mapfile reads LINES -> would keep "0 2" as one token)
EFF_GPUS=()
echo "BBQ-V background launcher"
echo "  requested GPUs: $GPUS   (min free ${MIN_FREE_MB} MiB, force=${FORCE_GPU})"
for g in "${GPU_REQUESTED[@]}"; do
  if [[ "$FORCE_GPU" == "1" ]]; then
    EFF_GPUS+=("$g"); echo "  gpu $g : FORCE_GPU=1 -> using (no VRAM check)"; continue
  fi
  free="$(gpu_free_mb "$g")"
  if [[ -z "$free" ]]; then
    echo "  gpu $g : no VRAM info (nvidia-smi missing / index?) -> USING anyway (verify manually!)"
    EFF_GPUS+=("$g"); continue
  fi
  if (( free < MIN_FREE_MB )); then
    echo "  gpu $g : only ${free} MiB free (< ${MIN_FREE_MB}) -> SKIPPING (would OOM)."
  else
    echo "  gpu $g : ${free} MiB free -> OK."
    EFF_GPUS+=("$g")
  fi
done
N_GPU="${#EFF_GPUS[@]}"
if (( N_GPU == 0 )); then
  echo "ERROR: no usable GPU (all below ${MIN_FREE_MB} MiB free). Set FORCE_GPU=1 or MIN_FREE_MB lower." >&2
  exit 1
fi

# ---- balanced shard computation over the usable GPUs ----------------------
base=$(( NUM_IMAGES / N_GPU ))
rem=$(( NUM_IMAGES % N_GPU ))
echo "  usable GPUs   : ${EFF_GPUS[*]}  (${N_GPU})"
echo "  images        : $NUM_IMAGES random rows  ->  ~${base}..$((base + 1)) per GPU ($(( base * N_GPU + rem )) accounted)"
echo "  steps each    : $STEPS"
echo "  seed          : ${SEED:-<unset> (driver default 0)}"
echo "  fresh         : $FRESH (1 = rewrite+reseed, 0 = keep)"
echo "  out / logs    : $OUT_DIR  /  $LOG_DIR"
if (( ! DRY_RUN )); then
  mkdir -p "$LOG_DIR"
  : > "$MASTER_LOG"        # truncate master once per launch
  : > "$PID_FILE"
fi

# ---- launch one detached driver per usable GPU ----------------------------
start=0
for i in "${!EFF_GPUS[@]}"; do
  gpu="${EFF_GPUS[$i]}"
  count=$(( base + (i < rem ? 1 : 0) ))
  if (( count <= 0 )); then continue; fi
  row_end=$(( start + count - 1 ))
  log="$LOG_DIR/gpu${gpu}.log"
  fresh_flag=""; [[ "$FRESH" == "1" ]] && fresh_flag="--fresh" || fresh_flag="--keep"

  # Build the per-GPU driver invocation as a safe array (no unquoted expansion).
  cmd=("$PY" "$HERE/run_bbq_loop.py"
       --start-index "$start" --num-images "$count" --steps "$STEPS"
       --gpu "$gpu" "$fresh_flag"
       --max-consecutive-failures "$MAX_CONSECUTIVE_FAILURES"
       --out "$OUT_DIR"
       --vision-model "$VISION_MODEL" --image-model "$IMAGE_MODEL"
       --width "$WIDTH" --height "$HEIGHT")
  if [[ -n "$SEED" ]]; then cmd+=(--seed "$SEED"); fi

  if (( DRY_RUN )); then
    echo "  [dry-run] gpu=$gpu  sel=$start..$row_end  n=$count   (log: $log)"
    printf '          would run:'
    for _x in "${cmd[@]}"; do printf ' %q' "$_x"; done
    echo
    start=$(( row_end + 1 )); continue
  fi

  {
    echo "==================== [$(date -Is)] gpu=$gpu sel=$start..$row_end n=$count steps=$STEPS ===================="
  } >> "$MASTER_LOG"
  nohup "${cmd[@]}" >> "$log" 2>&1 &
  pid=$!
  echo "$pid gpu=$gpu sel=$start..$row_end n=$count" >> "$PID_FILE"
  echo "  LAUNCHED gpu=$gpu  sel=$start..$row_end  n=$count  pid=$pid   (log: $log)"
  start=$(( row_end + 1 ))
done

if (( DRY_RUN )); then
  echo
  echo "(dry-run: nothing launched, no files written.)"
  exit 0
fi
echo
echo "--- launched ${N_GPU} job(s) ---"
busiest=$(( base + (rem > 0 ? 1 : 0) ))
eta_min=$(( busiest * 20 ))
printf 'per-GPU load    : ~%s images each; ETA ~%s min (~%.1f h) wall-clock, running in parallel\n' "$busiest" "$eta_min" "$(awk 'BEGIN{print '$eta_min'/60}')"
echo "master plan : $MASTER_LOG"
echo "per-GPU logs: ls $LOG_DIR/  (gpu0.log, gpu2.log, ...)"
echo "pid file    : $PID_FILE"
echo
echo "monitor  : content_convergence/run_bbq_200.sh --status   tail -f $LOG_DIR/gpu0.log   nvidia-smi"
echo "stop all : content_convergence/run_bbq_200.sh --stop"
