#!/usr/bin/env bash
# Resume the benchmark matrix across as many <=6h GPU jobs as it takes.
#
# WHY THIS EXISTS
#   The `normal` QoS caps a single job at 06:00:00 and this account at one GPU,
#   so the full matrix cannot be one `sbatch` call. Previously a cell killed at
#   the wall lost its entire jsonl (run_one wrote the file only at the end), so
#   a 6h chunk that overran threw away hours of work. This watcher fixes both:
#
#     * chunks     -- scripts/plan_resume.py groups the still-missing cells into
#                     chunks whose ESTIMATED GPU time stays under BUDGET
#                     (default 4.5h), comfortably inside the 6h wall.
#     * checkpoint -- each job runs `run_experiment.py --resume`, which skips
#                     cells already complete on disk and CONTINUES cells that
#                     were interrupted mid-way (raw records are flushed one
#                     sample at a time, so a kill keeps partial progress).
#     * re-plan    -- after every job the plan is rebuilt from disk, so a chunk
#                     that timed out never re-runs the cells it already did.
#
# USAGE (survives logout):
#   nohup setsid bash scripts/resume_benchmark.sh > logs/resume.out 2>&1 &
#
# TUNABLES (environment variables):
#   VARIANTS=baseline,improved   variants to complete        (default baseline,improved)
#   BUDGET=16200                 estimated seconds per chunk  (default 4.5h; must be <21600)
#   POLL=60                      seconds between queue checks (default 60)
#   MAX_ATTEMPTS=3               sbatch tries per chunk before abandoning it
#   MAX_WATCH_HOURS=72           hard stop so a stuck queue cannot spin forever
#
# DRY RUN (prints the plan, submits nothing):
#   DRY_RUN=1 bash scripts/resume_benchmark.sh
#
# Rebuild the final aggregate table afterwards (no GPU needed):
#   python scripts/rescore_raw.py --out final

set -uo pipefail

cd "$(dirname "$0")/.."

SLURM_BIN=/cm/local/apps/slurm/current/bin
[[ -d "$SLURM_BIN" ]] && export PATH="$SLURM_BIN:$PATH"

VARIANTS="${VARIANTS:-baseline,improved}"
BUDGET="${BUDGET:-16200}"
POLL="${POLL:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_WATCH_HOURS="${MAX_WATCH_HOURS:-72}"
DRY_RUN="${DRY_RUN:-0}"

# Refuse to plan a chunk that could hit the 6h wall.
if (( BUDGET >= 21600 )); then
  echo "BUDGET=${BUDGET}s is >= the 6h (21600s) QoS wall; refusing to run." >&2
  exit 2
fi

# The planner needs pyyaml, which lives in the project conda env.
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
if ! conda activate llm-assignment; then
  echo "cannot activate conda env 'llm-assignment'" >&2
  exit 2
fi

mkdir -p logs
export VARIANTS BUDGET

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# `squeue` is aliased to `squeue -la` in interactive shells, but aliases are not
# expanded in non-interactive scripts, so the plain binary is used here.
queue_lines() { squeue -u "$USER" -h -t PENDING,RUNNING -o '%i %j %T' 2>/dev/null; }
queue_count() { queue_lines | grep -c . || true; }
job_in_queue() { squeue -h -j "$1" -o '%i' 2>/dev/null | grep -q .; }

log "resume_benchmark started (pid $$, variants=$VARIANTS, budget=${BUDGET}s)"
log "queue now: $(queue_lines | tr '\n' ' ')"

declare -A attempts=()
SKIP_CELLS=""          # grows only when a chunk is abandoned
chunk_no=0
deadline=$(( $(date +%s) + MAX_WATCH_HOURS * 3600 ))

while true; do
  if (( $(date +%s) > deadline )); then
    log "watch deadline (${MAX_WATCH_HOURS}h) reached; exiting"
    exit 1
  fi

  # --- re-plan from disk ----------------------------------------------------
  if ! plan="$(SKIP_CELLS="$SKIP_CELLS" python scripts/plan_resume.py)"; then
    log "planner failed; exiting"
    exit 1
  fi

  if [[ -z "$plan" ]]; then
    log "all cells complete -- matrix is finished"
    break
  fi

  first="$(printf '%s\n' "$plan" | head -n1)"
  cells="$(printf '%s' "$first" | cut -f1)"
  cost="$(printf '%s' "$first" | cut -f2)"
  ncells="$(printf '%s' "$first" | cut -f3)"
  nchunks="$(printf '%s\n' "$plan" | grep -c . || true)"

  # --- serialise: never run two of our GPU jobs at once ---------------------
  inflight="$(queue_count)"
  if (( inflight > 0 )); then
    log "waiting: $inflight job(s) in queue, $nchunks chunk(s) planned"
    sleep "$POLL"
    continue
  fi

  if (( ${attempts["$cells"]:-0} >= MAX_ATTEMPTS )); then
    log "abandoning chunk after ${MAX_ATTEMPTS} attempts: $cells"
    SKIP_CELLS="${SKIP_CELLS:+$SKIP_CELLS,}$cells"
    continue
  fi

  chunk_no=$((chunk_no + 1))
  name="resume-${chunk_no}"

  if [[ "$DRY_RUN" != "0" ]]; then
    log "DRY_RUN: $nchunks chunk(s) planned; first would be $name ($ncells cell(s), ~$(( cost / 60 ))min)"
    log "DRY_RUN: sbatch --job-name=$name scripts/run_experiment.slurm --cells '$cells' --resume --scores-name resume_$chunk_no"
    break
  fi

  log "submitting $name: $ncells cell(s), ~$(( cost / 60 ))min"
  if ! submit_out="$(sbatch --job-name="$name" \
                            scripts/run_experiment.slurm \
                            --cells "$cells" \
                            --resume \
                            --scores-name "resume_${chunk_no}" 2>&1)"; then
    log "sbatch FAILED for $cells: $submit_out"
    sleep "$POLL"
    continue
  fi
  jobid="$(printf '%s' "$submit_out" | grep -oE '[0-9]+$' || true)"
  attempts["$cells"]=$(( ${attempts["$cells"]:-0} + 1 ))
  log "  $submit_out (attempt ${attempts["$cells"]})"

  if [[ -z "$jobid" ]]; then
    log "  could not parse job id; waiting one poll"
    sleep "$POLL"
    continue
  fi

  # Wait for the job to APPEAR first (avoids racing the scheduler into a
  # double submit), then wait for it to leave the queue.
  for _ in $(seq 1 60); do
    job_in_queue "$jobid" && break
    sleep 2
  done
  while job_in_queue "$jobid"; do
    sleep "$POLL"
  done
  log "$name ($jobid) finished; re-planning"
done

log "final raw files:"
find results -type f -path '*/raw/*.jsonl' -print 2>/dev/null | sed 's/^/  /' || log "  (none)"
IFS=',' read -r -a requested_variants <<< "$VARIANTS"
if [[ "$DRY_RUN" == "0" ]]; then
  for variant in "${requested_variants[@]}"; do
    raw_dir="results/${variant}/raw"
    [[ -d "$raw_dir" ]] || continue
    log "building ${variant} aggregate score table"
    python scripts/rescore_raw.py \
      --dir "$raw_dir" \
      --pattern "*__${variant}.jsonl" \
      --out "${variant}_final" \
      --out-dir "results/${variant}/scores"
  done
fi
log "resume_benchmark exiting"
