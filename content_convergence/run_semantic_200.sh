#!/usr/bin/env bash
#
# run_semantic_200.sh — background launcher for semantic_validate.py.
#
# Shards the run directories under OUT_DIR across the (free) GPUs, one embedding
# process per GPU, then produces the cross-run aggregate ONCE when the shards are
# done. Every job is detached (nohup) so it survives the shell exiting.
#
# Two phases, because this script's outputs are not all per-run:
#
#   1. SCORE   one process per GPU, each over its own slice of the runs, run with
#              --no-summary. The per-pair JSONL is named <run>.<kind>.sim.jsonl, one
#              file per run, so the shards never collide. The aggregate (summary CSV,
#              baseline, summary PNG) is suppressed here: each shard sees only its own
#              slice, so four shards would each overwrite the same files with a
#              partial, wrong answer.
#   2. COLLECT one pass over ALL runs once the shards exit. Every pair is already in
#              the JSONL by then, so nothing is re-embedded except the baseline
#              sample; it just reads, aggregates and plots. Launched automatically by
#              a detached waiter, or by hand with --collect.
#
# Usage (from /home/gpucce/Repos/content_convergence, or anywhere):
#   ./run_semantic_200.sh                          # all bbq_runs/image_*, both modes, 4 GPUs
#   GPUS="0 2" ./run_semantic_200.sh               # choose GPUs (space-separated)
#   MODE=images ./run_semantic_200.sh              # images only
#   ANCHOR=5 ./run_semantic_200.sh                 # drift from step 5 instead of the default 3
#   ANCHOR= ./run_semantic_200.sh                  # consecutive pairs only (no 2nd panel)
#   OUT_DIR=bbq_runs_smoke ./run_semantic_200.sh   # score the smoke run instead
#   PER_RUN_PLOTS=1 ./run_semantic_200.sh          # also one PNG per run (200 runs = 200 files)
#
# Subcommands:
#   ./run_semantic_200.sh --status                 # which tracked jobs are alive, + progress
#   ./run_semantic_200.sh --stop                   # SIGTERM (then SIGKILL) all jobs this script launched
#   ./run_semantic_200.sh --collect                # (re)build the aggregate from the JSONL on disk
#   ./run_semantic_200.sh --dry-run                # show the per-GPU plan and pair count; launch nothing
#
# Monitor:
#   tail -f bbq_runs/semantic/logs/gpu0.log        # per-GPU progress
#   tail -f bbq_runs/semantic/logs/collect.log     # the aggregate pass
#
# Env vars (all optional, shown with defaults):
#   OUT_DIR=bbq_runs MODE=both GPUS="0 1 2 3" STEP="" ANCHOR=3 \
#   IMAGE_MODEL=openai/clip-vit-large-patch14 TEXT_MODEL=intfloat/multilingual-e5-large \
#   MAX_SIZE=512 TEXT_MAX_LENGTH=512 BATCH_SIZE=16 BASELINE_POOL=48 SEED=0 \
#   PER_RUN_PLOTS=0 OVERWRITE=0 MIN_FREE_MB=8000 FORCE_GPU=0

set -euo pipefail

# ---- config (override via env) -------------------------------------------
OUT_DIR="${OUT_DIR:-bbq_runs}"                 # root holding the image_* run dirs
MODE="${MODE:-both}"                           # images | prompts | both
GPUS="${GPUS:-0 1 2 3}"
STEP="${STEP:-}"                               # blank = every consecutive pair
IMAGE_MODEL="${IMAGE_MODEL:-openai/clip-vit-large-patch14}"
TEXT_MODEL="${TEXT_MODEL:-intfloat/multilingual-e5-large}"
MAX_SIZE="${MAX_SIZE:-512}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# ${ANCHOR-3}, not ${ANCHOR:-3}: an explicitly EMPTY ANCHOR= means "off", while unset
# takes the default. Do not use 0 to disable -- step 0 is a legitimate anchor.
ANCHOR="${ANCHOR-1}"                           # also score every step vs this FIXED step
BASELINE_POOL="${BASELINE_POOL:-48}"           # runs sampled for the unrelated-pair floor
SEED="${SEED:-0}"
PER_RUN_PLOTS="${PER_RUN_PLOTS:-0}"
OVERWRITE="${OVERWRITE:-0}"                    # 1 = redo pairs already recorded
MIN_FREE_MB="${MIN_FREE_MB:-8000}"             # CLIP-L/14 + e5-large is ~2.5 GB; headroom for batches
FORCE_GPU="${FORCE_GPU:-0}"                    # 1 = skip the free-VRAM pre-flight

# One place decides the anchor flag, so the shards, the dry-run and the collect pass
# can never disagree about it -- a collect without it would drop the anchor records.
if [[ -n "$ANCHOR" ]]; then ANCHOR_FLAG=(--anchor "$ANCHOR"); else ANCHOR_FLAG=(--no-anchor); fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HERE/conda_venv/bin/python}"
VALIDATOR="$HERE/semantic_validate.py"

case "$OUT_DIR" in /*) RUNS_ROOT="$OUT_DIR" ;; *) RUNS_ROOT="$HERE/$OUT_DIR" ;; esac
SEM_DIR="$RUNS_ROOT/semantic"
LOG_DIR="$SEM_DIR/logs"
MASTER_LOG="$LOG_DIR/semantic_master.log"
COLLECT_LOG="$LOG_DIR/collect.log"
PID_FILE="$LOG_DIR/semantic.pids"

# ---- helpers -------------------------------------------------------------
gpu_free_mb() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr ',' ' ' \
    | awk -v i="$1" '$1==i { print $2-$3; exit }'
}

collect_all_runs() {
  shopt -s nullglob
  local runs=( "$RUNS_ROOT"/image_*/ )
  shopt -u nullglob
  (( ${#runs[@]} )) || { echo "no runs to collect under $RUNS_ROOT" >&2; return 1; }
  local cmd=( "$PY" "$VALIDATOR" "${runs[@]}"
              --mode "$MODE" --out "$SEM_DIR" --gpu "${1:-0}"
              --image-model "$IMAGE_MODEL" --text-model "$TEXT_MODEL"
              --max-size "$MAX_SIZE" --text-max-length "$TEXT_MAX_LENGTH"
              --batch-size "$BATCH_SIZE" --baseline-pool "$BASELINE_POOL" --seed "$SEED" )
  [[ -n "$STEP" ]] && cmd+=(--step "$STEP")
  # The collect pass must carry the SAME anchor: it rebuilds the plan, and without it
  # the anchor records on disk would be left out of the aggregate entirely.
  cmd+=("${ANCHOR_FLAG[@]}")
  "${cmd[@]}"
}

# ---- internal: wait for the scoring shards, then collect ------------------
# Invoked detached by the launcher; not part of the public interface.
if [[ "${1:-}" == "--_collect-after" ]]; then
  shift
  for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 10; done
  done
  echo "==================== [$(date -Is)] all shards exited; collecting ===================="
  collect_all_runs "${COLLECT_GPU:-0}"
  exit $?
fi

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
  written=$(cat "$SEM_DIR"/*.sim.jsonl 2>/dev/null | wc -l)
  files=$(ls "$SEM_DIR"/*.sim.jsonl 2>/dev/null | wc -l)
  echo "progress: $written pair(s) scored across $files file(s) under $SEM_DIR"
  if [[ -f "$SEM_DIR/semantic_summary.png" ]]; then
    echo "aggregate: $SEM_DIR/semantic_summary.png ($(date -r "$SEM_DIR/semantic_summary.png" -Is))"
  else
    echo "aggregate: not built yet (runs after the shards; see $COLLECT_LOG)"
  fi
  exit 0
fi

if [[ "${1:-}" == "--stop" ]]; then
  echo "stopping semantic run (pid file: $PID_FILE)"
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
  echo "note: the aggregate was not built -- run './run_semantic_200.sh --collect' when ready."
  exit 0
fi

if [[ "${1:-}" == "--collect" ]]; then
  read -ra _g <<< "$GPUS"
  echo "collecting the aggregate from the JSONL under $SEM_DIR (gpu ${_g[0]}) ..."
  collect_all_runs "${_g[0]}"
  exit $?
fi

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
read -ra GPU_REQUESTED <<< "$GPUS"
EFF_GPUS=()
echo "semantic_validate background launcher"
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
echo "  encoders      : images=$IMAGE_MODEL"
echo "                  prompts=$TEXT_MODEL"
echo "  baseline pool : $BASELINE_POOL run(s)   seed: $SEED   (0 = no unrelated-pair floor)"
echo "  per-run plots : $PER_RUN_PLOTS   overwrite: $OVERWRITE (0 = resume, skip recorded pairs)"
echo "  out / logs    : $SEM_DIR  /  $LOG_DIR"

# Shared flags for every scoring shard. --no-summary is the important one: see the
# header. The aggregate is built once, afterwards, by the collect phase.
COMMON=( --mode "$MODE" --out "$SEM_DIR" --no-summary
         --image-model "$IMAGE_MODEL" --text-model "$TEXT_MODEL"
         --max-size "$MAX_SIZE" --text-max-length "$TEXT_MAX_LENGTH"
         --batch-size "$BATCH_SIZE" )
[[ -n "$STEP" ]]             && COMMON+=(--step "$STEP")
COMMON+=("${ANCHOR_FLAG[@]}")
[[ "$PER_RUN_PLOTS" == "1" ]] && COMMON+=(--per-run-plots)
[[ "$OVERWRITE" == "1" ]]     && COMMON+=(--overwrite)

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
  : > "$MASTER_LOG"
  : > "$PID_FILE"
fi

# ---- phase 1: launch one detached scoring shard per usable GPU ------------
echo
PIDS=()
start=0
base=$(( N_RUNS / N_GPU ))
rem=$(( N_RUNS % N_GPU ))
for i in "${!EFF_GPUS[@]}"; do
  gpu="${EFF_GPUS[$i]}"
  count=$(( base + (i < rem ? 1 : 0) ))
  (( count > 0 )) || continue
  shard=( "${RUNS[@]:start:count}" )
  first="$(basename "${shard[0]}")"; last="$(basename "${shard[-1]}")"
  log="$LOG_DIR/gpu${gpu}.log"

  cmd=( "$PY" "$VALIDATOR" "${shard[@]}" "${COMMON[@]}" --gpu "$gpu" )

  if (( DRY_RUN )); then
    echo "  [dry-run] score  gpu=$gpu  n=$count  $first .. $last   (log: $log)"
    start=$(( start + count )); continue
  fi

  echo "==================== [$(date -Is)] gpu=$gpu n=$count $first..$last mode=$MODE ====================" >> "$MASTER_LOG"
  nohup "${cmd[@]}" >> "$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "$pid gpu=$gpu n=$count $first..$last" >> "$PID_FILE"
  echo "  LAUNCHED score  gpu=$gpu  n=$count  $first .. $last  pid=$pid   (log: $log)"
  start=$(( start + count ))
done

# ---- phase 2: a detached waiter builds the aggregate when the shards exit --
if (( DRY_RUN )); then
  echo "  [dry-run] collect gpu=${EFF_GPUS[0]}  over all $N_RUNS run(s) once the shards exit  (log: $COLLECT_LOG)"
  echo
  echo "(dry-run: nothing launched, no files written.)"
  exit 0
fi

: > "$COLLECT_LOG"
COLLECT_GPU="${EFF_GPUS[0]}" nohup "$HERE/$(basename "${BASH_SOURCE[0]}")" \
  --_collect-after "${PIDS[@]}" >> "$COLLECT_LOG" 2>&1 &
collect_pid=$!
echo "$collect_pid collect (waits for the shards, then aggregates)" >> "$PID_FILE"
echo "  LAUNCHED collect pid=$collect_pid  waits for ${#PIDS[@]} shard(s)   (log: $COLLECT_LOG)"

echo
echo "--- launched ${#PIDS[@]} scoring job(s) + 1 collector ---"
echo "master plan : $MASTER_LOG"
echo "per-GPU logs: ls $LOG_DIR/  (gpu0.log, gpu1.log, ...)"
echo "aggregate   : $SEM_DIR/semantic_summary.png  (written by the collector at the end)"
echo "pid file    : $PID_FILE"
echo
echo "monitor  : ./run_semantic_200.sh --status   tail -f $COLLECT_LOG"
echo "re-plot  : ./run_semantic_200.sh --collect"
echo "stop all : ./run_semantic_200.sh --stop"
