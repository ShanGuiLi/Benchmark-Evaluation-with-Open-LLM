"""Plan the remaining benchmark cells into chunks that fit the 6h QoS wall.

The ``normal`` QoS caps a single SLURM job at 06:00:00 and this account at one
GPU, so the full matrix cannot be one ``sbatch`` call. This planner:

  1. reads ``configs/models.yaml`` + ``configs/tasks.yaml`` to learn the matrix,
  2. checks ``results/raw/`` for cells that are missing or only *partially*
     written (a cell killed at the wall leaves a short file),
  3. estimates each remaining cell's runtime from the measured ``time_s``
     columns of ``results/scores/*.csv`` (falling back to the wildcard-variant
     measurement, i.e. the baseline time, for a variant never run before),
  4. packs them into chunks with a first-fit-decreasing heuristic so that every
     chunk's estimated GPU time stays under ``BUDGET`` seconds.

Biggest first matters: the long ``deepseek-r1-distill`` cells get their own
chunk instead of being paired with more work that would push the chunk over the
6h wall and get it killed mid-cell.

Stdout is one line per chunk, TAB-separated (so the caller can capture it
cleanly while diagnostics stream to stderr)::

    <model:task:variant,...>\t<estimated_seconds>\t<n_cells>

Environment:
    VARIANTS     comma-separated variants to consider (default baseline,improved)
    BUDGET       max estimated seconds per chunk (default 16200 = 4.5h; QoS cap 6h)
    LOAD_SEC     model-load overhead per distinct model in a chunk (default 120)
    SKIP_CELLS   comma-separated model:task:variant cells to ignore (abandoned)

Example::

    VARIANTS=baseline,improved python scripts/plan_resume.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import RESULTS_DIR  # noqa: E402
from src.config import load_models, load_tasks  # noqa: E402

# Fallback per-task sample counts (full split). Used only when no raw file for
# that task exists yet to measure the real count from.
DEFAULT_COUNTS = {"commonsense_qa": 710, "openbook_qa": 491, "piqa": 696}
# Fallback per-task runtime estimate (seconds) for a totally cold start.
DEFAULT_COST = {"commonsense_qa": 4000.0, "openbook_qa": 3000.0, "piqa": 4000.0}


def _env_list(name: str, default: str = "") -> list:
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def expected_count(task_key: str) -> int:
    """Largest record count seen for ``task_key`` across existing raw files.

    A completed variant gives the true full-split size, so this stays correct
    even if ``--limit`` was used; the constant is only a cold-start fallback.
    """
    best = 0
    for f in RESULTS_DIR.rglob(f"*__{task_key}__*.jsonl"):
        best = max(best, _count_lines(f))
    return best or DEFAULT_COUNTS.get(task_key, 0)


def load_measured_times() -> dict:
    """``(model, task, variant)`` and ``(model, task, '*')`` -> seconds.

    Scraped from the long-form score CSVs; pivot CSVs have no ``time_s`` column
    and are skipped naturally.
    """
    times: dict = {}
    for f in sorted(RESULTS_DIR.rglob("*.csv")):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    raw = (row.get("time_s") or "").strip()
                    if not raw:
                        continue
                    try:
                        t = float(raw)
                    except ValueError:
                        continue
                    if t <= 0:
                        continue
                    model = (row.get("model") or "").strip()
                    task = (row.get("task") or "").strip()
                    variant = (row.get("variant") or "").strip()
                    if not model or not task:
                        continue
                    key = (model, task, variant)
                    times[key] = max(times.get(key, 0.0), t)
                    wild = (model, task, "*")
                    times[wild] = max(times.get(wild, 0.0), t)
        except OSError:
            continue
    return times


def estimate(model: str, task_key: str, variant: str, times: dict) -> float:
    for key in ((model, task_key, variant), (model, task_key, "*")):
        if times.get(key):
            return times[key]
    return DEFAULT_COST.get(task_key, 3600.0)


def main() -> int:
    variants = _env_list("VARIANTS", "baseline,improved")
    budget = float(os.environ.get("BUDGET", "16200"))
    load_sec = float(os.environ.get("LOAD_SEC", "120"))

    skip = set()
    for raw in _env_list("SKIP_CELLS"):
        parts = raw.split(":")
        if len(parts) == 3:
            skip.add(tuple(p.strip() for p in parts))

    models = [m["key"] for m in load_models()["models"]]
    tasks = [(t["key"], t["name"]) for t in load_tasks()["tasks"]]
    counts = {key: expected_count(key) for key, _ in tasks}
    times = load_measured_times()

    missing = []
    for variant in variants:
        for model in models:
            for task_key, task_name in tasks:
                if (model, task_key, variant) in skip or (model, task_name, variant) in skip:
                    continue
                path = RESULTS_DIR / variant / "raw" / f"{model}__{task_key}__{variant}.jsonl"
                have = _count_lines(path) if path.exists() else 0
                need = counts.get(task_key, 0)
                if need and have >= need:
                    continue  # complete
                missing.append({
                    "model": model,
                    "task_key": task_key,
                    "task_name": task_name,
                    "variant": variant,
                    "have": have,
                    "need": need,
                    "cost": estimate(model, task_key, variant, times),
                })

    if not missing:
        print("[plan] all requested cells are complete", file=sys.stderr)
        return 0

    total = sum(c["cost"] for c in missing)
    print(f"[plan] {len(missing)} cell(s) missing, ~{total / 3600:.2f}h of estimated "
          f"GPU work, budget {budget / 3600:.2f}h per chunk", file=sys.stderr)

    # First-fit decreasing.
    missing.sort(key=lambda c: -c["cost"])
    chunks = []
    for cell in missing:
        for ch in chunks:
            extra = cell["cost"] + (load_sec if cell["model"] not in ch["models"] else 0.0)
            if ch["cost"] + extra <= budget:
                ch["cells"].append(cell)
                ch["models"].add(cell["model"])
                ch["cost"] += extra
                break
        else:
            chunks.append({
                "cells": [cell],
                "models": {cell["model"]},
                "cost": cell["cost"] + load_sec,
            })

    for i, ch in enumerate(chunks, start=1):
        flag = "  <-- OVER BUDGET: single cell cannot be split" if ch["cost"] > budget else ""
        print(f"[plan] chunk {i}: {len(ch['cells'])} cell(s), "
              f"{ch['cost'] / 3600:.2f}h{flag}", file=sys.stderr)
        for c in ch["cells"]:
            note = f"  (resume from {c['have']}/{c['need']})" if c["have"] else ""
            print(f"[plan]    - {c['model']}/{c['task_key']}/{c['variant']}"
                  f"  ~{c['cost'] / 60:.1f}min{note}", file=sys.stderr)
        cells_csv = ",".join(
            f'{c["model"]}:{c["task_name"]}:{c["variant"]}' for c in ch["cells"]
        )
        print(f"{cells_csv}\t{ch['cost']:.0f}\t{len(ch['cells'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
