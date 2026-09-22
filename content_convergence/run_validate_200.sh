#!/usr/bin/env bash
#
# run_validate_200.sh — background launcher for validate_run_diffs.py.
#
# Shards the run directories under OUT_DIR across the (free) GPUs and runs one
# validate_run_diffs.py PER GPU, each pinned to its GPU. Only the VLM is loaded
# (no diffusion model), so this needs far less VRAM than run_bbq_200.sh. Every
# job is detached (nohup) so it survives the shell exiting.
#
# Sharding is safe: the JSONL filename is <run>.<kind>.jsonl, one file per run
# directory, so the per-GPU processes never write to the same file even though
# they all share one --out.
#
# Re-running RESUMES: pairs already in the JSONL are skipped, so a shard that
# died can be picked up by re-launching the same command. OVERWRITE=1 redoes them.
#
# Usage (from the repo root, or anywhere):
#   content_convergence/run_validate_200.sh                        # all bbq_runs/image_*, both modes, 4 GPUs
#   GPUS="0 2" content_convergence/run_validate_200.sh             # choose GPUs (space-separated)
#   MODE=images content_convergence/run_validate_200.sh            # skip the prompt-side comparison (half the work)
#   ANCHOR=5 content_convergence/run_validate_200.sh               # anchor on step 5 instead of the default 3
#   ANCHOR= content_convergence/run_validate_200.sh                # consecutive pairs only (HALVES the work)
#   OUT_DIR=bbq_runs_smoke content_convergence/run_validate_200.sh # validate the smoke run instead
#   STEP=3 content_convergence/run_validate_200.sh                 # only the (3,4) pair of each run
#
# Subcommands:
#   content_convergence/run_validate_200.sh --status               # which tracked jobs are alive, + progress
#   content_convergence/run_validate_200.sh --stop                 # SIGTERM (then SIGKILL) all jobs this script launched
#   content_convergence/run_validate_200.sh --dry-run              # show the per-GPU plan and exact pair count; launch nothing
#
# Monitor:
#   tail -f bbq_runs/validation/logs/gpu0.log    # per-GPU progress
#   wc -l bbq_runs/validation/*.jsonl | tail -1  # comparisons written so far
#
# Env vars (all optional, shown with defaults):
#   OUT_DIR=bbq_runs MODE=both GPUS="0 1 2 3" STEP="" ANCHOR=3 \
#   VISION_MODEL=Qwen/Qwen3-VL-4B-Instruct MAX_SIZE=512 MAX_NEW_TOKENS=768 \
#   KEEP_RAW=0 OVERWRITE=0 MIN_FREE_MB=12000 FORCE_GPU=0

set -euo pipefail

# ---- config (override via env) -------------------------------------------
OUT_DIR="${OUT_DIR:-bbq_runs}"                 # root holding the image_* run dirs
MODE="${MODE:-both}"                           # images | prompts | both
GPUS="${GPUS:-0 1 2 3}"
STEP="${STEP:-}"                               # blank = every consecutive pair
# ${ANCHOR-3}, not ${ANCHOR:-3}: an explicitly EMPTY ANCHOR= means "off", while unset
# takes the default. Do not use 0 to disable -- step 0 is a legitimate anchor.
ANCHOR="${ANCHOR-1}"                           # also compare every step vs this FIXED step
                                               # (on by default; roughly DOUBLES the VLM calls)
VISION_MODEL="${VISION_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
MAX_SIZE="${MAX_SIZE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
KEEP_RAW="${KEEP_RAW:-0}"                      # 1 = store the model's raw reply too
OVERWRITE="${OVERWRITE:-0}"                    # 1 = redo pairs already recorded
MIN_FREE_MB="${MIN_FREE_MB:-12000}"            # VLM alone is ~9-10 GB; leave headroom
FORCE_GPU="${FORCE_GPU:-0}"                    # 1 = skip the free-VRAM pre-flight

# One place decides the anchor flag, so the shards and the dry-run cannot disagree.
if [[ -n "$ANCHOR" ]]; then ANCHOR_FLAG=(--anchor "$ANCHOR"); else ANCHOR_FLAG=(--no-anchor); fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # content_convergence/
ROOT="$(dirname "$HERE")"                              # repo root: conda_venv/, bbq_runs/, ...
PY="${PY:-$ROOT/conda_venv/bin/python}"
VALIDATOR="$HERE/validate_run_diffs.py"

# OUT_DIR may be relative to the repo root, like run_bbq_200.sh's own default.
case "$OUT_DIR" in /*) RUNS_ROOT="$OUT_DIR" ;; *) RUNS_ROOT="$ROOT/$OUT_DIR" ;; esac
VALID_DIR="$RUNS_ROOT/validation"
LOG_DIR="$VALID_DIR/logs"
MASTER_LOG="$LOG_DIR/validate_master.log"
PID_FILE="$LOG_DIR/validate.pids"

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
  written=$(cat "$VALID_DIR"/*.jsonl 2>/dev/null | wc -l)
  files=$(ls "$VALID_DIR"/*.jsonl 2>/dev/null | wc -l)
  echo "progress: $written comparison(s) across $files file(s) under $VALID_DIR"
  exit 0
fi

if [[ "${1:-}" == "--stop" ]]; then
  echo "stopping validation run (pid file: $PID_FILE)"
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
  echo "done. (partial JSONL is kept; re-launching resumes from it.)"
  exit 0
fi

# ---- dry-run: print the plan, launch nothing ------------------------------
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ---- collect the run directories ------------------------------------------
shopt -s nullglob
RUNS=( "$RUNS_ROOT"/image_*/ )
shopt -u nullglob
N_RUNS="${#RUNS[@]}"
if (( N_RUNS == 0 )); then
  echo "ERROR: no run directory matching $RUNS_ROOT/image_*/" >&2
  exit 1
fi

# ---- pre-flight: keep only GPUs with enough free VRAM ---------------------
read -ra GPU_REQUESTED <<< "$GPUS"   # word-split on IFS
EFF_GPUS=()
echo "validate_run_diffs background launcher"
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

echo "  usable GPUs   : ${EFF_GPUS[*]}  (${N_GPU})"
echo "  runs          : $N_RUNS under $RUNS_ROOT  ->  ~$(( N_RUNS / N_GPU )) per GPU"
echo "  mode          : $MODE   step: ${STEP:-<all consecutive pairs>}"
echo "  anchor        : ${ANCHOR:-<off> (consecutive pairs only)}"
echo "  model         : $VISION_MODEL  (max-size $MAX_SIZE, max-new-tokens $MAX_NEW_TOKENS)"
echo "  keep-raw      : $KEEP_RAW   overwrite: $OVERWRITE (0 = resume, skip recorded pairs)"
echo "  out / logs    : $VALID_DIR  /  $LOG_DIR"

# Shared flags for every shard.
COMMON=( --mode "$MODE" --out "$VALID_DIR"
         --vision-model "$VISION_MODEL"
         --max-size "$MAX_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" )
if [[ -n "$STEP" ]];        then COMMON+=(--step "$STEP"); fi
COMMON+=("${ANCHOR_FLAG[@]}")
if [[ "$KEEP_RAW" == "1" ]];  then COMMON+=(--keep-raw); fi
if [[ "$OVERWRITE" == "1" ]]; then COMMON+=(--overwrite); fi

if (( DRY_RUN )); then
  echo
  echo "  counting pairs (validator --dry-run over all runs; no model loaded) ..."
  # Capture rather than pipe: the validator prints every pair, and closing the
  # pipe early (head) would kill it with BrokenPipeError under `set -o pipefail`.
  plan="$("$PY" "$VALIDATOR" "${RUNS[@]}" --mode "$MODE" ${STEP:+--step "$STEP"} \
            "${ANCHOR_FLAG[@]}" --dry-run)"
  printf '%s\n' "$plan" | sed -n '1,2p' | sed 's/^/  /'
else
  mkdir -p "$LOG_DIR"
  : > "$MASTER_LOG"        # truncate master once per launch
  : > "$PID_FILE"
fi

# ---- launch one detached validator per usable GPU -------------------------
echo
start=0
base=$(( N_RUNS / N_GPU ))
rem=$(( N_RUNS % N_GPU ))
for i in "${!EFF_GPUS[@]}"; do
  gpu="${EFF_GPUS[$i]}"
  count=$(( base + (i < rem ? 1 : 0) ))
  if (( count <= 0 )); then continue; fi
  shard=( "${RUNS[@]:start:count}" )
  first="$(basename "${shard[0]}")"; last="$(basename "${shard[-1]}")"
  log="$LOG_DIR/gpu${gpu}.log"

  cmd=( "$PY" "$VALIDATOR" "${shard[@]}" "${COMMON[@]}" --gpu "$gpu" )

  if (( DRY_RUN )); then
    echo "  [dry-run] gpu=$gpu  n=$count  $first .. $last   (log: $log)"
    start=$(( start + count )); continue
  fi

  {
    echo "==================== [$(date -Is)] gpu=$gpu n=$count $first..$last mode=$MODE ===================="
  } >> "$MASTER_LOG"
  nohup "${cmd[@]}" >> "$log" 2>&1 &
  pid=$!
  echo "$pid gpu=$gpu n=$count $first..$last" >> "$PID_FILE"
  echo "  LAUNCHED gpu=$gpu  n=$count  $first .. $last  pid=$pid   (log: $log)"
  start=$(( start + count ))
done

if (( DRY_RUN )); then
  echo
  echo "(dry-run: nothing launched, no files written.)"
  exit 0
fi

echo
echo "--- launched ${N_GPU} job(s) ---"
echo "master plan : $MASTER_LOG"
echo "per-GPU logs: ls $LOG_DIR/  (gpu0.log, gpu1.log, ...)"
echo "pid file    : $PID_FILE"
echo
echo "monitor  : content_convergence/run_validate_200.sh --status   tail -f $LOG_DIR/gpu${EFF_GPUS[0]}.log"
echo "stop all : content_convergence/run_validate_200.sh --stop"
