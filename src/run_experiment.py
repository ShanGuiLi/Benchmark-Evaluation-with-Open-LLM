"""Run the benchmark experiment matrix: models x tasks x prompt variants.

This is the main entry point for the assignment. It evaluates every selected
model on every selected task with every selected prompt variant, writes the raw
per-run outputs and aggregates under ``results/<variant>/``.

Loop order: variant (outer) -> model -> task (inner).

Variants are staged deliberately: the baseline prompt is evaluated across ALL
models and tasks first, and only then is the improved prompt evaluated. This
means a staged workflow (run baseline, inspect, tune the prompt, run improved)
never mixes the two, and a job that dies midway still leaves a complete,
usable baseline matrix on disk.

Within one variant each model is loaded ONCE and reused across all tasks, so
model loads stay at (n_models x n_variants) instead of
(n_models x n_variants x n_tasks).

Examples
--------
    # stage 1: full baseline matrix (4 models x 3 tasks, baseline prompt)
    python src/run_experiment.py --variants baseline

    # stage 2: full improved matrix, run only after stage 1 is verified
    python src/run_experiment.py --variants improved

    # quick smoke test: 1 model, 1 task, 20 samples
    python src/run_experiment.py --models tinyllama-1.1b --tasks CommonsenseQA \
        --limit 20 --variants baseline

    # both stages back-to-back in one job (baseline completes first)
    python src/run_experiment.py --variants baseline,improved

On this cluster run it on a GPU node, e.g.:
    srun --partition=MGPU-TC2 --qos=normal --gres=gpu:1 --cpus-per-task=8 \
         --mem=32G --time=02:00:00 python src/run_experiment.py --variants baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import RESULTS_DIR  # noqa: E402
from src.config import load_models, load_tasks  # noqa: E402
from src.data_loader import load_grouped_datasets  # noqa: E402
from src.evaluate import evaluate_samples, summarise_records  # noqa: E402
from src.infer import LLM, pick_dtype  # noqa: E402
from src.prompt_templates import build_baseline_prompt, build_improved_prompt  # noqa: E402

# Prompt variants supported for the main experiment (few-shot handled later).
VARIANT_BUILDERS: Dict[str, Callable[[dict], str]] = {
    "baseline": build_baseline_prompt,
    "improved": build_improved_prompt,
}

# Models that emit long reasoning traces before the final answer need a much
# larger generation budget; with a tiny budget the answer is never reached and
# extraction returns nothing. Can be overridden per model in configs/models.yaml
# via ``max_new_tokens: <int>``.
#
# The baseline prompt does not constrain the output format at all, so the model
# is free to write a full sentence before naming a letter. A 16-token budget
# cut those sentences off mid-word, which made the baseline look far worse than
# it is and turned the baseline-vs-improved comparison into a format artifact.
# Both variants share one budget per model, so the comparison stays fair.
REASONING_MODEL_KEYS = {"deepseek-r1-distill-qwen-1.5b"}
DEFAULT_MAX_NEW_TOKENS = 64
REASONING_MAX_NEW_TOKENS = 1024


def _resolve_max_new_tokens(model_cfg: dict, global_default: int, cli_override: int | None) -> int:
    """Priority: CLI override > per-model config > reasoning heuristic > global."""
    if cli_override is not None:
        return cli_override
    if "max_new_tokens" in model_cfg:
        return int(model_cfg["max_new_tokens"])
    if model_cfg["key"] in REASONING_MODEL_KEYS:
        return REASONING_MAX_NEW_TOKENS
    return global_default


def _parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _load_records(path: Path) -> List[dict]:
    """Read back per-sample records from a (possibly partial) raw jsonl file."""
    if not path.exists():
        return []
    out: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _row_from_stats(
    stats: dict,
    model_key: str,
    task_key: str,
    variant: str,
    raw_path: Path,
    elapsed: float,
    resumed: bool,
) -> dict:
    """Build one summary row from extraction statistics."""
    return {
        "model": model_key,
        "task": task_key,
        "variant": variant,
        "n": stats["n"],
        "parsed": stats["parsed"],
        "correct": stats["correct"],
        "correct_parsed": stats["correct_parsed"],
        "accuracy": round(stats["accuracy"], 2),
        "accuracy_parsed": round(stats["accuracy_parsed"], 2),
        "unparsed": stats["unparsed"],
        "pred_top": stats["pred_top"],
        "pred_top_pct": round(stats["pred_top_share"] * 100, 1),
        "degenerate": stats["degenerate"],
        "time_s": round(elapsed, 1),
        "resumed": resumed,
        "raw_path": str(raw_path.relative_to(RESULTS_DIR)),
    }


def run_one(
    llm: LLM,
    samples: List[dict],
    build_prompt: Callable[[dict], str],
    model_key: str,
    task_key: str,
    variant: str,
    resume: bool = False,
) -> dict:
    """Evaluate one (model, task, variant) cell and persist its raw outputs.

    With ``resume`` the raw file is treated as a checkpoint:

    * all samples already present -> the cell is skipped (no GPU work) and the
      summary row is rebuilt straight from disk;
    * a prefix of the samples present -> generation continues from where the
      previous run stopped instead of starting over;
    * otherwise the cell runs fresh.

    New records are appended and flushed one sample at a time, so a wall-time
    kill keeps every sample generated so far (the earlier all-or-nothing write
    threw away a whole cell on a timeout).
    """
    raw_dir = RESULTS_DIR / variant / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{model_key}__{task_key}__{variant}.jsonl"

    existing: List[dict] = []
    if resume:
        existing = _load_records(raw_path)[: len(samples)]

    if existing and len(existing) >= len(samples):
        print(f"  [{model_key}] {task_key} / {variant}  ({len(samples)} samples) "
              f"-- already complete, skipping")
        return _row_from_stats(
            summarise_records(existing), model_key, task_key, variant,
            raw_path, 0.0, resumed=True,
        )

    def infer_fn(sample: dict) -> str:
        return llm.generate(build_prompt(sample))

    suffix = f"  (resuming at {len(existing)}/{len(samples)})" if existing else ""
    print(f"  [{model_key}] {task_key} / {variant}  ({len(samples)} samples){suffix} ...")
    start = time.time()
    with open(raw_path, "a" if existing else "w", encoding="utf-8") as fh:
        def sink(record: dict) -> None:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # survive a SIGKILL from the scheduler's wall-time limit

        _label_counts, _accuracy, records = evaluate_samples(
            samples, infer_fn, records=existing, on_record=sink,
        )
    elapsed = time.time() - start

    return _row_from_stats(
        summarise_records(records), model_key, task_key, variant,
        raw_path, elapsed, resumed=bool(existing),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark matrix")
    parser.add_argument("--models", default="all",
                        help="comma-separated model keys, or 'all'")
    parser.add_argument("--tasks", default="all",
                        help="comma-separated task names, or 'all'")
    parser.add_argument("--variants", default="baseline",
                        help="comma-separated prompt variants (baseline,improved)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap samples per task (0 = full split)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="force a generation budget for every model")
    parser.add_argument("--scores-name", default="summary",
                        help="output CSV base name in results/<variant>/scores/")
    parser.add_argument("--cells", default="",
                        help="explicit model:task:variant cells (comma-separated); "
                             "overrides --models/--tasks/--variants")
    parser.add_argument("--resume", action="store_true",
                        help="treat results/raw/*.jsonl as checkpoints: skip cells "
                                "already complete and continue cells interrupted mid-way")
    args = parser.parse_args()

    models_cfg = load_models()
    tasks_cfg = load_tasks()
    inference_cfg = models_cfg.get("inference", {})
    global_max_new = int(inference_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))

    # ----- select models / tasks / variants ---------------------------- #
    all_models = models_cfg["models"]
    model_by_key = {m["key"]: m for m in all_models}
    all_tasks = tasks_cfg["tasks"]
    task_by_name = {t["name"]: t for t in all_tasks}
    for t in all_tasks:  # accept the lowercase task key as an alias too
        task_by_name.setdefault(t["key"], t)

    def _pick_models(value: str) -> List[dict]:
        if value == "all":
            return all_models
        wanted = set(_parse_csv(value))
        picked = [m for m in all_models if m["key"] in wanted]
        missing = wanted - {m["key"] for m in picked}
        if missing:
            raise SystemExit(f"unknown model key(s): {sorted(missing)}")
        return picked

    def _pick_tasks(value: str) -> List[dict]:
        if value == "all":
            return all_tasks
        wanted = set(_parse_csv(value))
        picked = [t for t in all_tasks if t["name"] in wanted]
        missing = wanted - {t["name"] for t in picked}
        if missing:
            raise SystemExit(f"unknown task(s): {sorted(missing)}")
        return picked

    # ``plan[variant][model_key] = [task_name, ...]`` is the exact set of cells
    # to evaluate, in order. With --cells a chunked resume job runs ONLY the
    # cells it was handed; otherwise it is the full model x task rectangle.
    plan: Dict[str, Dict[str, List[str]]] = {}

    if args.cells:
        for raw_cell in _parse_csv(args.cells):
            parts = [p.strip() for p in raw_cell.split(":")]
            if len(parts) != 3:
                raise SystemExit(
                    f"bad --cells entry {raw_cell!r}; expected model:task:variant")
            mkey, tname, variant = parts
            if mkey not in model_by_key:
                raise SystemExit(f"unknown model key {mkey!r} in --cells")
            if tname not in task_by_name:
                raise SystemExit(f"unknown task {tname!r} in --cells")
            if variant not in VARIANT_BUILDERS:
                raise SystemExit(f"unknown variant {variant!r} in --cells")
            canonical = task_by_name[tname]["name"]
            tasks_for_model = plan.setdefault(variant, {}).setdefault(mkey, [])
            if canonical not in tasks_for_model:
                tasks_for_model.append(canonical)
        variants = list(plan)
    else:
        selected_models = _pick_models(args.models)
        selected_tasks = _pick_tasks(args.tasks)
        variants = _parse_csv(args.variants) or ["baseline"]
        for v in variants:
            if v not in VARIANT_BUILDERS:
                raise SystemExit(f"unknown variant {v!r}; choose from {sorted(VARIANT_BUILDERS)}")
        for v in variants:
            plan[v] = {
                m["key"]: [t["name"] for t in selected_tasks] for m in selected_models
            }

    # ----- load dataset ------------------------------------------------ #
    print("Loading benchmark ...")
    grouped = load_grouped_datasets(
        dataset_name=tasks_cfg["dataset"]["name"],
        config=tasks_cfg["dataset"]["config"],
        split=tasks_cfg["dataset"]["split"],
    )

    print("=" * 72)
    print(f"variants : {variants}")
    print(f"resume   : {args.resume}")
    print(f"limit    : {args.limit or 'full split'}")
    for v in variants:
        print(f"  [{v}] " + ", ".join(f"{m}:{len(ts)}" for m, ts in plan[v].items()))
    print("=" * 72)

    rows: List[dict] = []
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None

    # ----- main loops: variant (outer) -> model -> task (inner) -------- #
    # Baseline is fully evaluated before improved is started, so a staged
    # workflow always has a complete usable matrix for the earlier variant.
    for variant_idx, variant in enumerate(variants, start=1):
        print("\n" + "=" * 72)
        print(f"# VARIANT {variant_idx}/{len(variants)}: {variant}")
        print("=" * 72)

        for mkey, task_names in plan[variant].items():
            model_cfg = model_by_key[mkey]
            max_new = _resolve_max_new_tokens(model_cfg, global_max_new, args.max_new_tokens)
            print("\n" + "#" * 72)
            print(f"# MODEL: {model_cfg['name']}  ({model_cfg['hf_id']})")
            print(f"# variant={variant}  chat_template={model_cfg.get('chat_template', False)}  "
                  f"max_new_tokens={max_new}")
            print("#" * 72)

            llm = LLM(
                hf_id=model_cfg["hf_id"],
                use_chat_template=bool(model_cfg.get("chat_template", False)),
                dtype=pick_dtype(),
                max_new_tokens=max_new,
                do_sample=bool(inference_cfg.get("do_sample", False)),
                temperature=float(inference_cfg.get("temperature", 1.0)),
            )
            print(f"loaded on {llm.device}")

            for task_name in task_names:
                task_cfg = task_by_name[task_name]
                if task_name not in grouped:
                    print(f"  !! task {task_name!r} not in dataset, skipping")
                    continue
                samples = grouped[task_name]
                if args.limit:
                    samples = samples[: args.limit]

                row = run_one(
                    llm, samples, VARIANT_BUILDERS[variant],
                    model_cfg["key"], task_cfg["key"], variant, resume=args.resume,
                )
                rows.append(row)

            # free GPU memory before the next model
            del llm
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # flush results as soon as this variant finishes, so a later variant
        # (or a crash) can never cost us the completed matrix
        if rows:
            _write_summaries(rows, variants, args.scores_name)
            print(f"\n>>> variant {variant!r} complete; "
                f"summary written under {RESULTS_DIR / variant / 'scores'}")

    # ----- final report ------------------------------------------------ #
    if not rows:
        raise SystemExit("no results produced")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        for variant in variants:
            sub = df[df["variant"] == variant]
            if sub.empty:
                continue
            print(f"\n[{variant}]  accuracy % (all samples, unparsed counted wrong)")
            print(sub.pivot_table(index="model", columns="task", values="accuracy"))
            print(f"\n[{variant}]  accuracy % on parsed samples only")
            print(sub.pivot_table(index="model", columns="task",
                                  values="accuracy_parsed"))
            print(f"\n[{variant}]  unparsed outputs")
            print(sub.pivot_table(index="model", columns="task", values="unparsed"))

    bad = df[df["degenerate"]]
    if not bad.empty:
        print("\n" + "!" * 72)
        print("WARNING: degenerate predictors (one label dominates the parsed answers).")
        print("Their accuracy is an artifact of the label distribution, not skill:")
        for _, r in bad.iterrows():
            print(f"  {r['model']} / {r['task']} / {r['variant']}: "
                  f"{r['pred_top']!r} = {r['pred_top_pct']}% of parsed, "
                  f"acc={r['accuracy']}%")
        print("!" * 72)

    print(f"\nvariant summaries -> {RESULTS_DIR}/<variant>/scores/")


def _write_summaries(rows: List[dict], variants: List[str], scores_name: str) -> None:
    """Persist the long-form CSV plus one model x task pivot per variant.

    The pivot carries three blocks so a single file shows both the headline
    number and how trustworthy it is: ``accuracy`` (all samples, unparsed
    counted wrong), ``accuracy_parsed`` (parsed samples only) and ``unparsed``
    (count of outputs the extractor could not map to a label).
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    for variant in variants:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue
        scores_dir = RESULTS_DIR / variant / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(scores_dir / f"{scores_name}.csv", index=False)
        pivot = sub.pivot_table(
            index="model",
            columns="task",
            values=["accuracy", "accuracy_parsed", "unparsed"],
        )
        pivot.to_csv(scores_dir / f"{scores_name}__{variant}.csv")


if __name__ == "__main__":
    main()
