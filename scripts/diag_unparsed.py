"""Diagnostic: why are samples unparsed?

Splits the unparsed samples of a run into two buckets:
  * TRUNCATED  -- the generation looks cut off mid-sentence (does not end in a
                  sentence/closing punctuation and/or sits right at the token
                  budget). More tokens would plausibly help.
  * REFUSED    -- the generation ends cleanly but never commits to a letter.
                  More tokens will not help; the strict extractor is right.

Also reports the predicted-label distribution so degenerate (mode-collapsed)
models can be identified even when the strict parse rate is low.

Usage:
    python scripts/diag_unparsed.py --dir results/raw/_smoke64
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_tasks  # noqa: E402
from src.data_loader import load_grouped_datasets  # noqa: E402
from src.evaluate import extract_answer  # noqa: E402

CLEAN_END = tuple('.!?")]\'')


def build_index(grouped):
    idx = {}
    for samples in grouped.values():
        for s in samples:
            idx[s["question"]] = (
                [o["label"] for o in s["options"]],
                [o["text"] for o in s["options"]],
            )
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/raw")
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--show", type=int, default=2, help="unparsed samples to echo")
    args = ap.parse_args()

    tasks_cfg = load_tasks()
    grouped = load_grouped_datasets(
        dataset_name=tasks_cfg["dataset"]["name"],
        config=tasks_cfg["dataset"]["config"],
        split=tasks_cfg["dataset"]["split"],
    )
    idx = build_index(grouped)

    for path in sorted(glob.glob(os.path.join(args.dir, args.pattern))):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        parsed, trunc, refused = 0, 0, 0
        lens, lens_unparsed = [], []
        preds = Counter()
        examples = []
        for r in rows:
            text = r.get("raw_output") or ""
            labels, texts = idx.get(r.get("question", ""), ([], []))
            lens.append(len(text))
            pred = extract_answer(text, labels, texts)
            if pred:
                parsed += 1
                preds[pred] += 1
            else:
                lens_unparsed.append(len(text))
                clean = text.rstrip().endswith(CLEAN_END)
                # A clean ending but no committed letter => model refused.
                if clean:
                    refused += 1
                else:
                    trunc += 1
                if len(examples) < args.show:
                    examples.append(text[-140:].replace("\n", "\\n"))
        n = len(rows)
        top = preds.most_common(1)
        top_str = f"{top[0][0]}={top[0][1]/parsed*100:.0f}%" if parsed else "-"
        lens.sort()
        print(f"\n=== {os.path.basename(path)}")
        print(
            f"    n={n} parsed={parsed} ({parsed/n*100:.0f}%)  "
            f"unparsed={n-parsed}  -> truncated={trunc} refused={refused}"
        )
        print(
            f"    len all: min={lens[0]} med={lens[len(lens)//2]} max={lens[-1]}  "
            f"| len unparsed: med={sorted(lens_unparsed)[len(lens_unparsed)//2] if lens_unparsed else '-'}"
        )
        print(f"    pred distribution (parsed only): {dict(preds)}  top={top_str}")
        for e in examples:
            print(f"      UNPARSED end: ...{e}")


if __name__ == "__main__":
    main()
