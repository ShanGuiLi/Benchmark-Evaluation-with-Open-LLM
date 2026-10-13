"""Prompt templates for the benchmark tasks.

Three variants are provided:

* ``build_baseline_prompt``  - the original prompt from the example notebook
  (``Question`` + options + ``Answer:``), used as the baseline.
* ``build_improved_prompt``  - a constrained prompt that asks for a single
  option letter, which is what the benchmark's ``answerKey`` expects.
* ``build_fewshot_prompt``   - few-shot variant that prepends worked examples.

All builders keep the exact same option formatting so comparisons isolate the
effect of the prompt change itself.
"""

from __future__ import annotations

from typing import List, Optional


def _format_options(options: List[dict]) -> str:
    """Render the options block exactly like the example notebook."""
    return "".join(f"{opt['label']}: {opt['text']}\n" for opt in options)


def _option_letters(sample: dict) -> str:
    """Human-readable list of the sample's option labels, e.g. ``"A, B, C, D, E"``."""
    return ", ".join(opt["label"] for opt in sample["options"])


def _improved_instruction(sample: dict) -> str:
    """Instruction constrained to the letters that actually exist for this task.

    Note: options are NOT always A/B/C/D -- CommonsenseQA has 5 options (A-E),
    piqa has only 2 (A-B). We therefore build the letter list dynamically.
    """
    return (
        "Choose the best answer from the options above. "
        f"Output exactly one label from [{_option_letters(sample)}] and nothing else."
    )


def build_baseline_prompt(sample: dict) -> str:
    """Original prompt from the provided example code.

    ``Question ...\\nA: ...\\nB: ...\\nAnswer: ``
    """
    prompt = f"{sample['question']}\n"
    prompt += _format_options(sample["options"])
    prompt += "Answer: "
    return prompt


def build_improved_prompt(sample: dict) -> str:
    """Improved prompt requesting a single option letter."""
    prompt = f"{sample['question']}\n"
    prompt += _format_options(sample["options"])
    prompt += f"{_improved_instruction(sample)}\nAnswer: "
    return prompt


def build_fewshot_prompt(sample: dict, examples: List[dict]) -> str:
    """Few-shot prompt: prepend solved examples before the target question."""
    prompt = ""
    for ex in examples:
        prompt += f"Question: {ex['question']}\n"
        prompt += _format_options(ex["options"])
        prompt += f"Answer: {ex['answerKey']}\n\n"

    prompt += f"Question: {sample['question']}\n"
    prompt += _format_options(sample["options"])
    prompt += f"{_improved_instruction(sample)}\nAnswer: "
    return prompt


# name -> function(sample) (few-shot needs examples, handled separately)
BUILDERS = {
    "baseline": build_baseline_prompt,
    "improved": build_improved_prompt,
}


def build_prompt(variant: str, sample: dict, fewshot_examples: Optional[List[dict]] = None) -> str:
    """Dispatch to the requested prompt variant.

    ``variant`` is one of ``"baseline"``, ``"improved"``, ``"fewshot"``.
    """
    if variant == "fewshot":
        if not fewshot_examples:
            raise ValueError("fewshot variant requires fewshot_examples")
        return build_fewshot_prompt(sample, fewshot_examples)
    try:
        return BUILDERS[variant](sample)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown prompt variant: {variant!r}") from exc
