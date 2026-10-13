"""Diagnostic: is the strict extractor losing real signal?

For every saved ``results/raw/*.jsonl`` file we compare the current (strict)
extractor against several deliberately liberal reference extractors. If the
liberal rules pick up many *correct* answers that the strict rule rejects, the
strict rule is too strict and is understating accuracy. If they mostly pick up
*wrong* letters, the strict rule is behaving correctly and ``unparsed`` is an
honest statistic rather than lost signal.

Option labels/texts are not stored in the raw file, so they are rebuilt from
the cached benchmark dataset by matching on the question text.

Usage:
    python scripts/analyze_strictness.py [--dir results/raw] [--pattern '*.jsonl']
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_tasks  # noqa: E402
from src.data_loader import load_grouped_datasets  # noqa: E402
from src.evaluate import extract_answer  # noqa: E402

_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")
_ANSWER_RE = re.compile(
    r"answer\s*(?:is|:|-|=>|=)?\s*\(?([A-Za-z])\b", re.IGNORECASE
)


def lib_first_letter(text: str, valid: set) -> str:
    m = _LETTER_RE.search(text)
    return m.group(1) if m and m.group(1) in valid else ""


def lib_last_letter(text: str, valid: set) -> str:
    hits = [m.group(1) for m in _LETTER_RE.finditer(text) if m.group(1) in valid]
    return hits[-1] if hits else ""


def lib_tail_first(text: str, valid: set, window: int = 200) -> str:
    return lib_first_letter(text[-window:], valid)


def lib_answer_regex(text: str, valid: set) -> str:
    for m in _ANSWER_RE.finditer(text):
        if m.group(1).upper() in valid:
            return m.group(1).upper()
    return ""


def lib_majority(text: str, valid: set) -> str:
    """Most frequently mentioned option letter -- a 'vote' style upper bound."""
    counts: Dict[str, int] = {}
    for m in _LETTER_RE.finditer(text):
        if m.group(1) in valid:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


LIBERAL = {
    "first_letter": lib_first_letter,
    "last_letter": lib_last_letter,
    "tail_first": lib_tail_first,
    "answer_regex": lib_answer_regex,
    "majority": lib_majority,
}


def build_option_index(grouped: Dict[str, List[dict]]) -> Dict[str, Tuple[List[str], List[str]]]:
    index: Dict[str, Tuple[List[str], List[str]]] = {}
    for samples in grouped.values():
        for s in samples:
            index[s["question"]] = (
                [opt["label"] for opt in s["options"]],
                [opt["text"] for opt in s["options"]],
            )
    return index


def analyse_file(path: str, index: Dict[str, Tuple[List[str], List[str]]]) -> Dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    strict_correct = strict_parsed = n_unparsed = missing = 0
    lib_correct = {k: 0 for k in LIBERAL}
    recov_ok = {k: 0 for k in LIBERAL}
    recov_bad = {k: 0 for k in LIBERAL}

    for r in rows:
        text = r.get("raw_output", "") or ""
        gold = r.get("gold", "")
        pair = index.get(r.get("question", ""))
        if pair is None:
            missing += 1
            continue
        labels, texts = pair
        valid = {str(x).upper() for x in labels}

        pred = extract_answer(text, labels, texts)
        if pred:
            strict_parsed += 1
            if pred == gold:
                strict_correct += 1
        else:
            n_unparsed += 1

        for name, fn in LIBERAL.items():
            p = fn(text, valid)
            if p == gold:
                lib_correct[name] += 1
                if not pred:
                    recov_ok[name] += 1
            elif p and not pred:
                recov_bad[name] += 1

    n = len(rows) - missing
    return {
        "n": n, "missing": missing,
        "strict_parsed": strict_parsed,
        "strict_acc": strict_correct / n * 100 if n else 0.0,
        "unparsed": n_unparsed,
        "lib_acc": {k: (v / n * 100 if n else 0.0) for k, v in lib_correct.items()},
        "recov_ok": recov_ok, "recov_bad": recov_bad,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/raw")
    ap.add_argument("--pattern", default="*.jsonl")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no files matched {args.dir}/{args.pattern}")

    tasks_cfg = load_tasks()
    grouped = load_grouped_datasets(
        dataset_name=tasks_cfg["dataset"]["name"],
        config=tasks_cfg["dataset"]["config"],
        split=tasks_cfg["dataset"]["split"],
    )
    index = build_option_index(grouped)

    hdr = (
        f"{'file':56s} {'n':>3s} {'parsed':>6s} {'acc':>6s} | "
        f"{'best_lib':>8s} {'gain':>6s} | {'recov_ok':>8s} {'recov_bad':>9s} {'rule':>13s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for p in paths:
        res = analyse_file(p, index)
        rule, lib_acc = max(res["lib_acc"].items(), key=lambda kv: kv[1])
        print(
            f"{os.path.basename(p)[:56]:56s} {res['n']:3d} {res['strict_parsed']:6d} "
            f"{res['strict_acc']:6.1f} | {lib_acc:8.1f} {lib_acc - res['strict_acc']:+6.1f} | "
            f"{res['recov_ok'][rule]:8d} {res['recov_bad'][rule]:9d} {rule:>13s}"
        )

    print(
        "\n'best_lib' = best accuracy any liberal rule reaches on the SAME\n"
        "generations; 'gain' = what the strict extractor might be leaving behind.\n"
        "'recov_ok'/'recov_bad' = among the currently-unparsed samples, how many\n"
        "the liberal rule would score right vs wrong. recov_bad >> recov_ok means\n"
        "the strict rule is correctly refusing to guess."
    )


if __name__ == "__main__":
    main()
