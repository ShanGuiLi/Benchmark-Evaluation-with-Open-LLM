#!/usr/bin/env bash
# Keep the full baseline matrix running as a chain of <=6h GPU jobs.
#
# WHY A CHAIN
#   The `normal` QoS on this cluster allows the account only
#       MaxSubmitPU=2  (at most TWO submitted jobs at a time)
#       MaxTRESPU     gres/gpu=1  (at most ONE GPU in total, so jobs serialise)
#       MaxWall       06:00:00    (a single job may not run longer than 6h)
#   The full baseline (4 models x 3 tasks x 1 variant = 12 cells, 1897 samples
#   per cell) is roughly 13h of serial GPU time, so it cannot be one sbatch
#   call. This watcher splits it into chunks that each fit comfortably inside
#   the 6h ceiling and submits the next chunk as soon as a queue slot frees, so
#   the GPU never idles between chunks.
#
#   The 3 fast models (64/64/192 new tokens) are grouped into one ~3.5h chunk.
#   deepseek-r1-distill emits a long chain of thought and needs 1024 tokens, so
#   it is split one task per chunk (~2.4h-3.5h each) instead of risking a
#   mid-cell timeout at the 6h wall (a killed cell loses its whole jsonl).
#
# USAGE (survives logout):
#   nohup setsid bash scripts/chain_baseline.sh > logs/chain_baseline.out 2>&1 &
#
# SAFE TO RE-RUN
#   A chunk is only submitted when ALL of these hold:
#     * none of its raw output files exist yet, and
#     * no job named "bench-<chunk>" is currently PENDING/RUNNING for this user.
#   The second condition is what stops the watcher from re-submitting a chunk
#   that is already sitting in the queue. Each chunk gets at most MAX_ATTEMPTS
#   submissions before it is abandoned (so a job that dies without producing
#   output is retried once, then reported).

set -uo pipefail

cd "$(dirname "$0")/.."

SLURM_BIN=/cm/local/apps/slurm/current/bin
[[ -d "$SLURM_BIN" ]] && export PATH="$SLURM_BIN:$PATH"

MAX_JOBS="${MAX_JOBS:-2}"          # MaxSubmitPU for the `normal` QoS
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"  # sbatch tries per chunk before it is abandoned
POLL="${POLL:-60}"                 # seconds between queue checks
MAX_WATCH_HOURS="${MAX_WATCH_HOURS:-36}"  # hard stop so a stuck queue cannot spin

# name | model keys | task names (CLI) | task keys (raw filenames)
chunks=(
  "fast3|tinyllama-1.1b,qwen2.5-3b-instruct,qwen3-4b-instruct-2507|CommonsenseQA,OpenbookQA,piqa|commonsense_qa,openbook_qa,piqa"
  "ds_cs|deepseek-r1-distill-qwen-1.5b|CommonsenseQA|commonsense_qa"
  "ds_ob|deepseek-r1-distill-qwen-1.5b|OpenbookQA|openbook_qa"
  "ds_piqa|deepseek-r1-distill-qwen-1.5b|piqa|piqa"
)

declare -A attempts=()

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# 0 when every raw file this chunk should produce already exists.
chunk_done() {
  local models="$1" task_keys="$2" m t
  while IFS= read -r m; do
    while IFS= read -r t; do
      [[ -f "results/raw/${m}__${t}__baseline.jsonl" ]] || return 1
    done < <(tr ',' '\n' <<< "$task_keys")
  done < <(tr ',' '\n' <<< "$models")
  return 0
}

log "chain_baseline started (pid $$, max $MAX_JOBS jobs, poll ${POLL}s)"
log "queue now: $(squeue -u "$USER" -h -o '%i:%j:%T' | tr '\n' ' ')"

deadline=$(( $(date +%s) + MAX_WATCH_HOURS * 3600 ))

while true; do
  if (( $(date +%s) > deadline )); then
    log "watch deadline (${MAX_WATCH_HOURS}h) reached, exiting"
    exit 1
  fi

  # Snapshot the queue once per poll. `%j` is the job name; our chunks are
  # named "bench-<chunk>", so membership in this list means "already in flight".
  queued_names=$(squeue -u "$USER" -h -t PENDING,RUNNING -o '%j')
  queued_count=$(squeue -u "$USER" -h -t PENDING,RUNNING -o '%i' | wc -l)

  missing=0      # chunks whose outputs are not all on disk yet
  inflight=0     # ... of those, how many already have a job queued
  todo=()        # ... and how many are ready to submit right now

  for spec in "${chunks[@]}"; do
    IFS='|' read -r name models tnames tkeys <<< "$spec"
    chunk_done "$models" "$tkeys" && continue
    missing=$((missing + 1))

    if grep -qx "bench-${name}" <<< "$queued_names"; then
      inflight=$((inflight + 1))
      continue
    fi
    (( ${attempts[$name]:-0} < MAX_ATTEMPTS )) && todo+=("$spec")
  done

  if (( missing == 0 )); then
    log "all chunks complete -- baseline matrix is finished"
    break
  fi
  if (( inflight == 0 && ${#todo[@]} == 0 )); then
    log "$missing chunk(s) still incomplete but none queued and none submittable"
    log "(each has been tried MAX_ATTEMPTS=$MAX_ATTEMPTS times -- check logs/bench-*.err)"
    exit 1
  fi

  if (( queued_count < MAX_JOBS && ${#todo[@]} > 0 )); then
    IFS='|' read -r name models tnames tkeys <<< "${todo[0]}"
    log "submitting chunk '$name' (queue=$queued_count/$MAX_JOBS, $inflight in flight, ${#todo[@]} ready)"
    # Everything after the script path is forwarded verbatim to
    # run_experiment.py, so --scores-name must come last, not as an sbatch
    # option. A per-chunk CSV name keeps the chunks from overwriting each
    # other's summary file; the final table is rebuilt from results/raw with
    # scripts/rescore_raw.py.
    if sbatch --job-name="bench-$name" \
              scripts/run_experiment.slurm \
              --models "$models" \
              --tasks "$tnames" \
              --variants baseline \
              --scores-name "base_${name}"; then
      attempts[$name]=$(( ${attempts[$name]:-0} + 1 ))
      log "  chunk '$name' submitted (attempt ${attempts[$name]})"
    else
      log "  sbatch FAILED for chunk '$name'"
    fi
  else
    log "waiting: queue=$queued_count/$MAX_JOBS, $inflight in flight, ${#todo[@]} ready"
  fi

  sleep "$POLL"
done

log "final raw files:"
ls -1 results/raw/*.jsonl 2>/dev/null | sed 's/^/  /' || log "  (none)"
log "chain_baseline exiting"
