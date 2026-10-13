"""Answer extraction and accuracy evaluation.

The benchmark stores the gold answer as an option label in ``answerKey``
(e.g. ``"A"``). Model outputs are free-form, so we apply a small, well-defined
cascade of rules to map the generated text back to a label. The rules are kept
simple and transparent so the report can describe them precisely.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import tqdm


def get_labels(sample: dict) -> List[str]:
    """Return the valid option labels for a sample (e.g. ['A','B','C','D'])."""
    return [opt["label"] for opt in sample["options"]]


def get_option_texts(sample: dict) -> List[str]:
    """Return the option texts, aligned with :func:`get_labels`."""
    return [opt["text"] for opt in sample["options"]]


# Closing tags emitted by reasoning models. Written with explicit unicode
# escapes so the source stays robust regardless of editor / encoding quirks:
#   </think>                                  plain Qwen / DeepSeek style
#   <U+FF5C"/"U+FF5C"thinking"U+FF5C">        full-width Qwen variant
#   <U+FF5C"end"U+2581"of"U+2581"thinking"U+FF5C">   DeepSeek-R1 variant
_BAR = "\uff5c"    # ｜  fullwidth vertical line
_BLANK = "\u2581"  # ▁  lower one eighth block

_THINK_END_RE = re.compile(
    r"</think(?:ing)?>"
    + "|" + "<" + _BAR + "/" + _BAR + "thinking" + _BAR + ">"
    + "|" + "<" + _BAR + "end" + _BLANK + "of" + _BLANK + "thinking" + _BAR + ">",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Drop a leading reasoning block if the model emitted one.

    DeepSeek-R1 / Qwen-style models wrap their chain-of-thought in thinking
    tags. Only the text after the *last* closing tag is kept; when no closing
    tag is present (e.g. generation was truncated mid-thought) the original text
    is returned so the downstream extraction rules still get a chance.
    """
    matches = list(_THINK_END_RE.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    return text.strip()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# Long generations (a chain of thought, or a base model continuing the option
# list) mention option letters all over the text. For those, only the closing
# passage can plausibly hold the *answer*, so the standalone-letter rule is
# restricted to a trailing window.
_LONG_OUTPUT_CHARS = 400
_TAIL_WINDOW_CHARS = 200

# Predicted-label concentration above which we call the model degenerate.
# The share is measured over *parsed* samples only, so it needs a minimum
# sample count to mean anything -- with fewer than 10 parsed samples "always
# answered A" is not yet evidence of mode collapse. The 0.80 share (rather than
# a stricter 0.90) is what catches a base model that answers A for ~7 of every
# 8 parsed items.
_DEGENERATE_MIN_PARSED = 10
_DEGENERATE_TOP_SHARE = 0.80

# Below this parsed fraction the reported accuracy is dominated by "the model
# did not produce a readable answer" rather than by being wrong.
_LOW_PARSE_SHARE = 0.50

# The article "A" is never directly followed by a finite form of "to be", so
# "A is ..." must be the option label A rather than the article.
_COPULA_RE = re.compile(r"\s*(?:is|are|was|were)\b", re.IGNORECASE)


def _unique_match_letter(
    pattern: str, text: str, valid: set, flags: int = 0
) -> str:
    """Return the single distinct option letter matched by ``pattern``.

    Returns ``""`` when the pattern matches nothing valid *or* when it matches
    different letters at different positions. Taking the first match in the
    ambiguous case is exactly how the old extractor invented answers out of
    text such as ``"Option A is a marsh, option B is a tundra."``, so an
    ambiguous match is treated as no match and the next rule gets a turn.
    """
    letters = {
        m.group(1).upper()
        for m in re.finditer(pattern, text, flags)
        if m.group(1).upper() in valid
    }
    return next(iter(letters)) if len(letters) == 1 else ""


def _unique_choice_letter(text: str, valid: set) -> str:
    """Rule 3: find "choose B" / "select option C" / "choice: D".

    Like :func:`_unique_match_letter` this refuses to guess when several
    different letters are named, but it additionally ignores mentions that a
    comparison marker introduces, because those are discussion of distractors
    rather than a commitment.
    """
    letters = set()
    for m in _CHOICE_RE.finditer(text):
        letter = m.group(1).upper()
        if letter not in valid:
            continue
        if _COMPARE_RE.search(text[: m.start()]):
            continue
        letters.add(letter)
    return next(iter(letters)) if len(letters) == 1 else ""


# One standalone capital letter, not glued to other letters.
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")

# A letter introduced by one of these is explicitly *not* the answer, so it
# must not make the surrounding output look ambiguous (or look like a match).
_NEGATED_RE = re.compile(r"\b(?:not|nor|no|neither|never)\s*$", re.IGNORECASE)

# "the answer is X" / "answer is: X" / "the best answer among the given
# options is B: ..." -- the copula may be several words away from "answer", so
# a short non-newline gap is allowed. The trailing guard rejects a letter that
# is part of a list ("the answer is: A and D"), which is genuinely ambiguous.
# ``answer\b`` (not ``answer``) keeps the plural "answers" from matching.
_ANSWER_STMT_RE = re.compile(
    r"answer\b(?:[^\n]{0,40}?\bis\b)?\s*[:\-=>]*\s*\(?([A-Za-z])\b\)?"
    r"(?!\s*(?:and|or|,)\s)",
    re.IGNORECASE,
)

# "choose B" / "select option C" / "choice: D".
_CHOICE_RE = re.compile(
    r"\b(?:choose|select|pick|go\s+with|opt(?:\s*for)?|option|choice)\b"
    r"\s*(?:is\s*)?[:\-]?\s*\(?([A-Za-z])\b",
    re.IGNORECASE,
)

# The noun forms ("option A") are common when the model is *discussing* a
# distractor -- "Similar to option A, while lizards can live in rocky areas..."
# -- and crediting such a mention invents an answer the model never gave (it
# biases every rambling baseline trace towards option A). A mention introduced
# by a comparison marker is therefore ignored.
_COMPARE_RE = re.compile(
    r"\b(?:similar|compared|unlike|other|than|aside|versus|vs|instead|before)\b"
    r"[^.\n]{0,40}$",
    re.IGNORECASE,
)


def _letter_candidates(text: str, valid: set) -> List[Tuple[str, int]]:
    """Standalone option letters that could be the answer, with offsets.

    Excludes the English article "A" ("A marsh is where fog forms") but keeps
    "A" when it is directly followed by a form of "to be", which an article can
    never be. Letters the model explicitly ruled out ("not A") are dropped too.
    """
    found: List[Tuple[str, int]] = []
    for m in _LETTER_RE.finditer(text):
        letter = m.group(1)
        if letter not in valid:
            continue
        if _NEGATED_RE.search(text[:m.start()]):
            continue
        rest = text[m.end():]
        if re.match(r"\s*[a-z]", rest) and not _COPULA_RE.match(rest):
            continue  # the article "A" starting an English phrase
        found.append((letter, m.start()))
    return found


def _mentioned_letters(text: str, valid: set) -> List[str]:
    """Every standalone option letter, including ones a negation ruled out.

    Feeds the ambiguity check: if the model mentions a second option letter
    anywhere in the same passage, we cannot know which one it settled on.
    """
    return [
        m.group(1)
        for m in _LETTER_RE.finditer(text)
        if m.group(1) in valid and not _NEGATED_RE.search(text[:m.start()])
    ]


def extract_answer(
    text: str,
    labels: List[str],
    option_texts: Optional[List[str]] = None,
) -> str:
    """Map a free-form generation to one of ``labels``.

    Rules are applied in order and stop at the first confident match:

      0. the whole output is a single letter (possibly wrapped in punctuation)
      1. ``\\boxed{X}``
      2. an explicit ``answer ... X`` statement, where the copula may be a few
         words away from "answer" (``the best answer among the given options is
         B``) and may be followed by a colon
      3. ``choose / select / pick / option / choice X`` -- unless the mention is
         part of a comparison ("Similar to option A, ..."), which is discussion
         of a distractor rather than a commitment
      4. a parenthesised or bracketed letter, e.g. ``(A)`` or ``[B]``

    Rules 1--4 fire only when they match exactly one distinct option letter;
    rule 5 requires the standalone letter to be unique as well. Ambiguity is
    never resolved by guessing -- see :func:`_unique_match_letter`.

      5. a **unique** standalone uppercase letter that does not start an
         English phrase -- the uniqueness requirement is what stops
         enumeration artifacts such as ``"1. A. 2. B. 3. C."`` from being
         read as the answer "A"
      6. an option *text* echoed verbatim in the output (helps completion-style
         models that repeat the option instead of naming its letter)

    Returns ``""`` when no rule matches. Ambiguous outputs deliberately return
    ``""`` instead of guessing: mis-parsed samples are far more damaging to the
    reported accuracy than unparsed ones, which stay visible as a separate
    statistic.
    """
    if not text:
        return ""

    valid = {str(lab).upper() for lab in labels}
    cleaned = strip_reasoning(text)

    # 0. Whole output is just a letter, possibly wrapped like "(A)".
    core = cleaned.strip().strip(".:;,?!()[]{}\"'`* \n\t")
    if len(core) == 1 and core.upper() in valid:
        return core.upper()

    # 1. \boxed{X}
    letter = _unique_match_letter(r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}", cleaned, valid)
    if letter:
        return letter

    # 2. "the answer is X" / "answer: X" / "the best answer ... is X"
    letter = _unique_match_letter(
        _ANSWER_STMT_RE.pattern, cleaned, valid, re.IGNORECASE,
    )
    if letter:
        return letter

    # 3. "choose B" / "option C" / "choice: D"
    letter = _unique_choice_letter(cleaned, valid)
    if letter:
        return letter

    # 4. Parenthesised / bracketed letter.
    letter = _unique_match_letter(r"[\(\[]\s*([A-Za-z])\s*[\)\]]", cleaned, valid)
    if letter:
        return letter

    # 5. A unique standalone uppercase letter, skipping the English article
    #    "A". Two conditions must hold: exactly one letter looks like a
    #    candidate, AND the passage mentions no other option letter. Without
    #    the second condition the rule is fooled by base models that merely
    #    continue the option list ("1. A. 2. B. 3. C.") and by reasoning models
    #    that discuss several options, and it manufactures an answer the model
    #    never gave. For long outputs only the closing passage is inspected.
    scope = (
        cleaned if len(cleaned) <= _LONG_OUTPUT_CHARS
        else cleaned[-_TAIL_WINDOW_CHARS:]
    )
    candidates = _letter_candidates(scope, valid)
    mentioned = set(_mentioned_letters(scope, valid))
    if len({letter for letter, _ in candidates}) == 1 and len(mentioned) == 1:
        return candidates[0][0]

    # 6. Option text echoed verbatim (conservative: unique match, len >= 3).
    if option_texts:
        haystack = _normalise(cleaned)
        hits = []
        for lab, opt_text in zip(labels, option_texts):
            needle = _normalise(opt_text or "")
            if len(needle) < 3:
                continue
            if re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack):
                hits.append(str(lab).upper())
        if len(hits) == 1 and hits[0] in valid:
            return hits[0]

    return ""


def evaluate_samples(
    samples: List[dict],
    infer_fn: Callable[[dict], str],
    show_progress: bool = True,
    records: Optional[List[dict]] = None,
    on_record: Optional[Callable[[dict], None]] = None,
) -> Tuple[Dict[str, int], float, List[dict]]:
    """Evaluate ``infer_fn`` over ``samples``.

    ``infer_fn`` receives a sample and returns the raw generated text.
    Returns ``(predictions, accuracy, records)`` where ``predictions`` counts
    each predicted label (with ``""`` for unparsed outputs), ``accuracy`` is a
    percentage over *all* samples, and ``records`` holds per-sample details for
    the raw results file.

    Unparsed samples count as wrong. Because that conflates "the model was
    wrong" with "we could not read the model's answer", the parsed-subset
    accuracy and the predicted-label distribution are printed too; see
    :func:`summarise_records` for the same statistics in the result rows.

    Checkpoint support: pass ``records`` holding the per-sample records already
    evaluated by a previous (interrupted) run and only the *remaining* samples
    are generated. Every freshly produced record is handed to ``on_record``
    immediately, so the caller can append it to disk and a wall-time kill keeps
    everything generated so far. Statistics still cover *all* samples.
    """
    records = records if records is not None else []
    done = len(records)
    correct = sum(1 for r in records if r["correct"])
    total = len(samples)
    label_counts: Dict[str, int] = {}
    for prior in records:
        label_counts[prior["predicted"]] = label_counts.get(prior["predicted"], 0) + 1

    iterator = tqdm.tqdm(
        samples[done:],
        desc="eval",
        total=total,
        initial=done,
        disable=not show_progress,
    )
    start = time.time()
    for sample in iterator:
        generated = infer_fn(sample)
        labels = get_labels(sample)
        option_texts = get_option_texts(sample)
        predicted = extract_answer(generated, labels, option_texts)
        gold = sample["answerKey"]

        is_correct = predicted == gold
        correct += int(is_correct)
        label_counts[predicted] = label_counts.get(predicted, 0) + 1

        record = {
            "question": sample["question"],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "raw_output": generated,
        }
        records.append(record)
        if on_record is not None:
            on_record(record)
    elapsed = time.time() - start

    accuracy = correct / total * 100 if total else 0.0
    stats = summarise_records(records)
    line = (
        f"  acc={accuracy:.2f}%  ({correct}/{total})"
        f"  parsed={stats['parsed']}/{stats['n']}"
        f"  acc_parsed={stats['accuracy_parsed']:.2f}%"
        f"  time={elapsed:.1f}s"
    )
    print(line)
    if stats["degenerate"]:
        print(
            f"  !! WARNING: degenerate predictions -- {stats['pred_top']!r} is "
            f"{stats['pred_top_share'] * 100:.1f}% of all parsed answers; "
            f"accuracy here is close to a label-frequency artifact, not real "
            f"signal"
        )
    if stats["low_parse"]:
        print(
            f"  !! WARNING: only {stats['parsed']}/{stats['n']} outputs were "
            f"parseable; the accuracy above is dominated by unreadable answers "
            f"and should be reported alongside the parse rate, not on its own"
        )
    return label_counts, accuracy, records


def summarise_records(records: List[dict]) -> dict:
    """Derive parsing statistics from per-sample evaluation records.

    Kept separate from :func:`evaluate_samples` so the same numbers can be
    recomputed offline from a saved ``results/raw/*.jsonl`` file.
    """
    n = len(records)
    parsed_records = [r for r in records if r["predicted"]]
    parsed = len(parsed_records)
    correct = sum(1 for r in records if r["correct"])
    correct_parsed = sum(1 for r in parsed_records if r["correct"])

    top_label = ""
    top_share = 0.0
    if parsed:
        counts: Dict[str, int] = {}
        for r in parsed_records:
            counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1
        top_label = max(counts, key=counts.get)
        top_share = counts[top_label] / parsed

    return {
        "n": n,
        "parsed": parsed,
        "unparsed": n - parsed,
        "correct": correct,
        "correct_parsed": correct_parsed,
        "accuracy": correct / n * 100 if n else 0.0,
        "accuracy_parsed": correct_parsed / parsed * 100 if parsed else 0.0,
        "pred_top": top_label,
        "pred_top_share": top_share,
        # A model that answers the same letter for ~every parsed sample is not
        # being evaluated, it is riding the label distribution; flag it loudly.
        "degenerate": (
            parsed >= _DEGENERATE_MIN_PARSED
            and top_share >= _DEGENERATE_TOP_SHARE
        ),
        # Below this parse rate the accuracy figure mostly measures "we could
        # not read the output" rather than model correctness.
        "low_parse": n > 0 and (parsed / n) < _LOW_PARSE_SHARE,
    }
