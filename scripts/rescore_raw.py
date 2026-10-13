"""Re-score saved ``results/raw/*.jsonl`` files with the current extractor.

Generation is the expensive part; extraction is not. Whenever
``src/evaluate.py`` changes, the raw outputs on disk can be re-scored offline
in a second, with no GPU, so fixing the answer extractor never costs a re-run.

The raw records store the question, the gold label and the generated text, but
not the option set, so the option labels/texts are rebuilt from the benchmark
dataset (a local, cached HF dataset) by matching on the question text.

Every file is also scored with a deliberately naive extractor that takes the
*first* option letter it sees -- the behaviour that used to be in production.
The gap between the two quantifies how much the old rule was inflating the
numbers.

Run:
    python scripts/rescore_raw.py                     # all files, printed table
    python scripts/rescore_raw.py --out rescored      # also write results/scores/
    python scripts/rescore_raw.py --pattern '*tinyllama*'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import RAW_DIR, RESULTS_DIR, SCORES_DIR  # noqa: E402
from src.config import load_tasks  # noqa: E402
from src.data_loader import load_grouped_datasets  # noqa: E402
from src.evaluate import extract_answer, summarise_records  # noqa: E402

_NAIVE_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def naive_extract(text: str, labels: List[str]) -> str:
    """The old behaviour: the first standalone option letter wins.

    Used only as a reference point for how inflated the old scores were.
    """
    valid = {str(lab).upper() for lab in labels}
    core = (text or "").strip().strip(".:;,?!()[]{}\"'`* \n\t")
    if len(core) == 1 and core.upper() in valid:
        return core.upper()
    for m in _NAIVE_LETTER_RE.finditer(text or ""):
        if m.group(1) in valid:
            return m.group(1)
    return ""


def build_option_index(
    grouped: Dict[str, List[dict]]
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Map question text -> (labels, option_texts) for every task."""
    index: Dict[str, Tuple[List[str], List[str]]] = {}
    for samples in grouped.values():
        for s in samples:
            labels = [opt["label"] for opt in s["options"]]
            texts = [opt["text"] for opt in s["options"]]
            index[s["question"]] = (labels, texts)
    return index


def rescore_file(
    path: Path, option_index: Dict[str, Tuple[List[str], List[str]]]
) -> dict:
    """Re-extract every record in ``path`` and compare with the naive rule."""
    records: List[dict] = []
    missing = 0
    naive_correct = 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj["question"]
            gold = obj["gold"]
            raw = obj.get("raw_output", "")

            labels, texts = option_index.get(question, ([], []))
            if not labels:
                missing += 1
                labels, texts = ["A", "B", "C", "D", "E"], []

            predicted = extract_answer(raw, labels, texts)
            records.append(
                {
                    "question": question,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": predicted == gold,
                    "raw_output": raw,
                }
            )
            naive_correct += int(naive_extract(raw, labels) == gold)

    stats = summarise_records(records)
    model, _, rest = path.stem.partition("__")
    task, _, variant = rest.partition("__")
    naive_acc = naive_correct / stats["n"] * 100 if stats["n"] else 0.0
    return {
        "model": model,
        "task": task,
        "variant": variant,
        "n": stats["n"],
        "parsed": stats["parsed"],
        "unparsed": stats["unparsed"],
        "correct": stats["correct"],
        "accuracy": round(stats["accuracy"], 2),
        "accuracy_parsed": round(stats["accuracy_parsed"], 2),
        "first_letter_acc": round(naive_acc, 2),
        "extractor_delta": round(stats["accuracy"] - naive_acc, 2),
        "pred_top": stats["pred_top"],
        "pred_top_pct": round(stats["pred_top_share"] * 100, 1),
        "degenerate": stats["degenerate"],
        "unknown_questions": missing,
        "raw_path": str(path.relative_to(RESULTS_DIR)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score saved benchmark JSONL files with the current extractor"
    )
    parser.add_argument("--dir", type=Path, default=RAW_DIR,
                        help="directory containing JSONL files (default: results/raw)")
    parser.add_argument("--pattern", default="*.jsonl",
                        help="glob applied inside --dir (default: *.jsonl)")
    parser.add_argument("--out", default=None,
                        help="base name for a CSV in the matching scores directory")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="summary directory (default: results/<variant>/scores for variant raw dirs)")
    args = parser.parse_args()

    raw_dir = args.dir.resolve()
    paths = sorted(raw_dir.glob(args.pattern))
    if not paths:
        print(f"no files matched {args.pattern!r} in {raw_dir}")
        return 1

    tasks_cfg = load_tasks()
    print("Loading benchmark to rebuild option sets ...")
    grouped = load_grouped_datasets(
        dataset_name=tasks_cfg["dataset"]["name"],
        config=tasks_cfg["dataset"]["config"],
        split=tasks_cfg["dataset"]["split"],
    )
    option_index = build_option_index(grouped)

    rows = []
    for p in paths:
        row = rescore_file(p, option_index)
        rows.append(row)
        if row["unknown_questions"]:
            print(f"  !! {p.name}: {row['unknown_questions']} questions not found "
                  f"in the dataset; those records used a fallback label set")

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 240, "display.max_columns", 60):
        print()
        print(df[["model", "task", "variant", "n", "parsed", "unparsed",
                  "accuracy", "accuracy_parsed", "first_letter_acc",
                  "extractor_delta", "pred_top", "pred_top_pct",
                  "degenerate"]].to_string(index=False))

    bad = df[df["degenerate"]]
    if not bad.empty:
        print("\nDegenerate predictors (one label >= 90% of parsed answers) --")
        print("their accuracy tracks the label distribution, not model skill:")
        for _, r in bad.iterrows():
            print(f"  {r['model']} / {r['task']} / {r['variant']}: "
                  f"{r['pred_top']!r} = {r['pred_top_pct']}%, acc={r['accuracy']}%")

    if args.out:
        out_dir = args.out_dir or (
            raw_dir.parent / "scores" if raw_dir.name == "raw" else SCORES_DIR
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{args.out}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nwritten -> {out_csv}")

    print("\nnote: 'first_letter_acc' is a deliberately naive first-letter reference")
    print("      (not an exact replica of the previous implementation), used only to")
    print("      show how much the old-style rule was inflating reported scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
