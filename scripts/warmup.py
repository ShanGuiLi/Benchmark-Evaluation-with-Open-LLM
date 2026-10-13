"""Warm-up: reproduce the provided example (``open_llm_benchmark.py``).

This is a faithful adaptation of the uploaded example script. The only changes
are the ones required to run it on this HPC cluster:

  * removed the Colab ``!pip install datasets`` and ``!huggingface-cli login``
    lines (deps are managed by the conda env ``llm-assignment``),
  * added an optional ``--limit`` for a quick smoke test,
  * saved the predictions to ``results/raw/`` for later inspection.

Everything else mirrors the example: the same dataset, model, prompt and
answer-comparison logic.

Flow (identical to the example):
  1. load ``Open-Style/Open-LLM-Benchmark`` ("questions") and group by task
  2. load the model with ``AutoTokenizer`` / ``AutoModelForCausalLM``
  3. ``infer_llm`` builds ``Question ... A: ... Answer: `` and generates
  4. ``evaluate_samples`` computes accuracy against ``sample["answerKey"]``

Run (on a GPU node via SLURM -- see ``scripts/warmup.slurm``):
    python scripts/warmup.py
    python scripts/warmup.py --limit 20        # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets  # noqa: E402
import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src import RAW_DIR  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Load and group the evaluation dataset  (verbatim from the example)
# --------------------------------------------------------------------------- #
eval_set = datasets.load_dataset("Open-Style/Open-LLM-Benchmark", "questions")
grouped_datasets = {}
for example in eval_set["train"]:
    dataset = example["dataset"]
    if dataset not in grouped_datasets:
        grouped_datasets[dataset] = []
    grouped_datasets[dataset].append(example)


# --------------------------------------------------------------------------- #
# 2. Load the model + tokenizer  (verbatim from the example)
# --------------------------------------------------------------------------- #
def load_model(model_id: str = "TinyLlama/TinyLlama_v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return tokenizer, model


def make_infer_llm(tokenizer, model):
    """Return the example's ``infer_llm`` bound to the loaded model."""

    device = model.device

    def infer_llm(sample):
        question = sample["question"]
        options = sample["options"]
        prompt = f"{question}\n"
        for option in options:
            prompt += f"{option['label']}: {option['text']}\n"

        # The example uses the bare "Answer: " prompt; this is the baseline
        # we will later improve in the prompt-engineering step.
        prompt += "Answer: "

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs["input_ids"],
            max_length=len(inputs["input_ids"][0]) + 5,
            num_return_sequences=1,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded_output.split("Answer:")[-1].strip()
        return answer

    return infer_llm


# --------------------------------------------------------------------------- #
# 3. Evaluate  (verbatim from the example)
# --------------------------------------------------------------------------- #
def evaluate_samples(samples, infer_llm):
    correct = 0
    total = len(samples)
    predictions = []

    s = time.time()
    for sample in tqdm.tqdm(samples):
        predicted_answer = infer_llm(sample)
        predictions.append(predicted_answer)

        if predicted_answer == sample["answerKey"]:
            correct += 1
    e = time.time()
    print("Time:", e - s)
    accuracy = correct / total * 100
    return predictions, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-up: reproduce the example")
    parser.add_argument("--model", default="TinyLlama/TinyLlama_v1.1",
                        help="HF model id (default matches the example)")
    parser.add_argument("--task", default="CommonsenseQA",
                        help="task name (key of grouped_datasets)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap number of samples (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"available tasks: {sorted(grouped_datasets.keys())}")
    print(f"model : {args.model}")
    print(f"task  : {args.task}")
    print("=" * 60)

    if args.task not in grouped_datasets:
        raise SystemExit(f"task {args.task!r} not found")

    samples = grouped_datasets[args.task]
    if args.limit:
        samples = samples[: args.limit]
    print(f"{args.task}: {len(samples)} samples")

    print("\nLoading model (first run downloads weights to the HF cache) ...")
    tokenizer, model = load_model(args.model)
    print(f"model on {model.device}")

    infer_llm = make_infer_llm(tokenizer, model)

    predictions, accuracy = evaluate_samples(samples, infer_llm)

    for i, sample in enumerate(samples):
        print(f"Question: {sample['question']}")
        print(f"Predicted Answer: {predictions[i]}")
        print(f"Correct Answer: {sample['answerKey']}\n")

    print(f"Accuracy: {accuracy:.2f}%")

    # Persist raw predictions for later inspection.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"warmup__{args.task.lower()}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for sample, pred in zip(samples, predictions):
            fh.write(json.dumps(
                {"question": sample["question"], "gold": sample["answerKey"],
                 "predicted": pred, "correct": pred == sample["answerKey"]},
                ensure_ascii=False,
            ) + "\n")
    print(f"raw results -> {out}")


if __name__ == "__main__":
    main()
