"""Data loading utilities for the Open-LLM-Benchmark.

The benchmark ships as a single ``questions`` split where each example carries a
``dataset`` field identifying the originating task (CommonsenseQA, OpenbookQA,
PiQA, ...). We group examples by that field so each task can be evaluated
independently.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import datasets

from src.config import load_tasks


def load_grouped_datasets(
    dataset_name: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """Load the benchmark and group examples by their originating task.

    The dataset coordinates default to the ones declared in
    ``configs/tasks.yaml`` so there is a single source of truth.

    Returns a dict mapping task name (e.g. ``"CommonsenseQA"``) to a list of
    example dicts. Each example looks like::

        {
          "question": "...",
          "options": [{"label": "A", "text": "..."}, ...],
          "answerKey": "A",
          "dataset": "CommonsenseQA",
          ...
        }
    """
    if dataset_name is None or config is None or split is None:
        ds_cfg = load_tasks()["dataset"]
        dataset_name = dataset_name or ds_cfg["name"]
        config = config or ds_cfg["config"]
        split = split or ds_cfg["split"]

    eval_set = datasets.load_dataset(dataset_name, config)

    grouped: Dict[str, List[dict]] = {}
    for example in eval_set[split]:
        task = example["dataset"]
        grouped.setdefault(task, []).append(example)

    return grouped


def list_available_tasks(grouped: Dict[str, List[dict]]) -> List[str]:
    """Return the sorted task names present in a grouped dataset."""
    return sorted(grouped.keys())


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    grouped = load_grouped_datasets()
    for name in list_available_tasks(grouped):
        print(f"{name}: {len(grouped[name])} examples")
