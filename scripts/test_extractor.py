"""Regression tests for src.evaluate.extract_answer.

Run:  python scripts/test_extractor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import extract_answer, strip_reasoning, summarise_records  # noqa: E402

# (generation, labels, option_texts, expected)  -- expected None == "don't care"
CASES = [
    # --- the bug that motivated this whole fix -------------------------------
    # article "A" must NOT be picked when no option text matches it
    ("A marsh is where fog forms", ["A", "B", "C", "D"],
     ["a tundra", "the plains", "a desert", "a swamp"], ""),
    ("A bank is correct", ["A", "B", "C", "D", "E"], None, ""),
    ("Aardvark is unrelated", ["A", "B"], None, ""),
    # ... but when the text genuinely echoes option A, crediting A is correct
    ("A marsh is where fog forms", ["A", "B", "C", "D"],
     ["a marsh", "a tundra", "the plains", "a desert"], "A"),
    # --- explicit answer markers --------------------------------------------
    ("The answer is C", ["A", "B", "C", "D"], None, "C"),
    ("Answer: D", ["A", "B", "C", "D"], None, "D"),
    ("The correct option is D.", ["A", "B", "C", "D"], None, "D"),
    ("answer is not A but B", ["A", "B"], None, "B"),
    # --- choice verbs --------------------------------------------------------
    ("I would choose B because it is right", ["A", "B"], None, "B"),
    ("I will select option C", ["A", "B", "C"], None, "C"),
    # --- bare letters --------------------------------------------------------
    ("E", ["A", "B", "C", "D", "E"], None, "E"),
    ("(B)", ["A", "B", "C", "D"], None, "B"),
    ("**A**", ["A", "B", "C"], None, "A"),
    ("A.", ["A", "B", "C"], None, "A"),
    ("A: bank", ["A", "B", "C", "D", "E"], None, "A"),
    (r"\boxed{C}", ["A", "B", "C", "D"], None, "C"),
    # --- option-text echo ----------------------------------------------------
    ("Department store", ["A", "B", "C", "D", "E"],
     ["bank", "library", "department store", "mall", "new york"], "C"),
    # --- must stay unparsed --------------------------------------------------
    ("3, 6,", ["A", "B", "C", "D", "E"], None, ""),
    ("I have no idea", ["A", "B"], None, ""),
    # --- ambiguous option-text echo must not guess ---------------------------
    ("it is either a bank or a library", ["A", "B"], ["bank", "library"], ""),
    # single lowercase letter is normalised to an option label
    ("a", ["A", "B"], ["a", "a"], "A"),

    # --- AMBIGUITY: enumeration artifacts must NOT be read as an answer ------
    # Regression: rule 5 used to return the FIRST standalone capital letter, so
    # a base model continuing the option list ("1. A. 2. B. 3. C.") was scored
    # as if it had answered "A". This is the bug that inflated TinyLlama's
    # scores and skewed DeepSeek's truncated reasoning traces.
    ("1. A. 2. B. 3. C. 4", ["A", "B"], None, ""),
    ("1. A. 2. B. 3. C. 4", ["A", "B", "C", "D"], None, ""),
    ("1. A\nAnswer: 2. B\nAnswer: 3.", ["A", "B"], None, ""),
    ("1. A\nA: a marsh\nB: a tundra", ["A", "B"], None, ""),
    ("A and B are both plausible", ["A", "B"], None, ""),
    ("Neither A nor B fits", ["A", "B"], None, ""),
    # a letter the model ruled out does not create ambiguity
    ("It is not C. The answer is B", ["A", "B", "C"], None, "B"),
    # several options named, no conclusion -> refuse to guess
    ("Option A is a marsh, option B is a tundra.", ["A", "B"], None, ""),
    # --- a genuine answer still wins ----------------------------------------
    ("Option A is a marsh, option B is a tundra. The answer is B.",
     ["A", "B"], None, "B"),
    # --- "A is ..." cannot be the article: it is the option label -----------
    ("A is the correct answer", ["A", "B"], None, "A"),
    ("A is a marsh", ["A", "B"], None, "A"),
    # --- long output: early letters are ignored, the tail decides -----------
    ("A. " + "filler text here. " * 30 + "C", ["A", "C"], None, "C"),
    # ... but a long output that names no letter in its tail is ambiguous
    ("A. B. " + "filler text here. " * 30, ["A", "B"], None, ""),
]

THINKING_CASES = [
    (" thinking blah blah  The answer is C.", ["A", "B", "C", "D"], "C"),
    # DeepSeek full-width closing tag: <\uff5cend\u2581of\u2581thinking\uff5c>
    ("reasoning <" + "\uff5c" + "end" + "\u2581" + "of" + "\u2581"
     + "thinking" + "\uff5c" + ">B", ["A", "B"], "B"),
    ("long thoughts without a closing tag", ["A", "B"], ""),
]


def _record(predicted: str, gold: str = "A") -> dict:
    return {"question": "q", "gold": gold, "predicted": predicted,
            "correct": predicted == gold, "raw_output": ""}


STATS_CASES = [
    # name, records, expected accuracy_parsed, expected degenerate
    ("all parsed, half right", [_record("A")] * 5 + [_record("B")] * 5, 50.0, False),
    # unparsed samples are excluded from accuracy_parsed but counted in accuracy
    ("half unparsed", [_record("A")] * 5 + [_record("")] * 5, 100.0, False),
    # a model that always answers the same letter is not being evaluated
    ("constant predictor", [_record("A")] * 20, 100.0, True),
]


def main() -> int:
    failures = 0
    print("== extract_answer ==")
    for gen, labels, texts, expected in CASES:
        got = extract_answer(gen, labels, texts)
        if expected is None:
            status = "skip"
        elif got == expected:
            status = "ok"
        else:
            status = "FAIL"
            failures += 1
        print(f"  [{status:>4}] {gen!r:45} labels={labels} -> {got!r} (want {expected!r})")

    print("\n== strip_reasoning / thinking tags ==")
    for gen, labels, expected in THINKING_CASES:
        stripped = strip_reasoning(gen)
        got = extract_answer(gen, labels)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{status:>4}] {gen!r:50} -> stripped={stripped!r} ans={got!r} (want {expected!r})")

    print("\n== summarise_records ==")
    for name, records, want_parsed_acc, want_degenerate in STATS_CASES:
        stats = summarise_records(records)
        ok = (
            abs(stats["accuracy_parsed"] - want_parsed_acc) < 1e-9
            and stats["degenerate"] is want_degenerate
        )
        status = "ok" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  [{status:>4}] {name:22} n={stats['n']} parsed={stats['parsed']} "
            f"acc={stats['accuracy']:.1f} acc_parsed={stats['accuracy_parsed']:.1f} "
            f"top={stats['pred_top']!r} degenerate={stats['degenerate']} "
            f"(want acc_parsed={want_parsed_acc} degenerate={want_degenerate})"
        )

    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
