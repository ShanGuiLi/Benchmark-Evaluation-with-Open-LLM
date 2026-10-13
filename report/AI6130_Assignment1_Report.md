# AI6130 | Assignment 1

## Benchmark Evaluation with Open LLMs

### Baseline Evaluation and Prompt Engineering

**Student:** [Your name]  
**Student ID:** [Your student ID]  
**Submission date:** 13 October 2026

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

## 1. Objective and Experimental Setup

This experiment evaluates four language models on CommonsenseQA, OpenbookQA, and PiQA from Open-Style/Open-LLM-Benchmark. Each model is tested with the original open-ended prompt and a constrained-answer prompt. The benchmark contains 710, 491, and 696 items for these tasks, respectively (1,897 items per model and prompt variant).

| Model | Input format | Maximum new tokens |
|---|---|---:|
| TinyLlama/TinyLlama_v1.1 | Plain text | 64 |
| Qwen/Qwen2.5-3B-Instruct | Chat template | 64 |
| DeepSeek-R1-Distill-Qwen-1.5B | Chat template | 1,024 |
| Qwen3-4B-Instruct-2507 | Chat template | 192 |

All models use greedy decoding (`do_sample=False`). Model-specific token budgets accommodate their different output styles; each budget is unchanged between that model's baseline and improved runs. Chat templates are used for the instruction-tuned chat models, while TinyLlama receives a plain-text completion prompt.

### Prompt Design

The baseline follows the provided example: the question, labeled options, and an open-ended `Answer:` cue. The improved prompt preserves the same question and option text, then requests exactly one valid option label and no other text. The allowed labels are generated for each item (A-E for CommonsenseQA, A-D for OpenbookQA, and A-B for PiQA). This changes answer-format guidance without adding examples or task-specific facts. No few-shot examples were used in the reported runs.

## 2. Results

Accuracy is the number of correct answers divided by all items; an unparsed output counts as incorrect. Parse rate is the proportion of outputs mapped to a valid option label. The table reports accuracy and parse rate as `accuracy / parse rate`; the final column is the change in overall accuracy in percentage points (pp). Values may differ slightly due to rounding.

| Model | Task | Baseline acc. / parse | Improved acc. / parse | Δ acc. |
|---|---|---:|---:|---:|
| TinyLlama-1.1B | CommonsenseQA | 2.11% / 9.4% | 0.14% / 1.7% | -1.97 pp |
|  | OpenbookQA | 2.85% / 11.6% | 0.20% / 1.2% | -2.65 pp |
|  | PiQA | 16.52% / 34.5% | 5.03% / 9.1% | -11.49 pp |
| Qwen2.5-3B-Instruct | CommonsenseQA | 63.10% / 88.5% | 77.75% / 100.0% | +14.65 pp |
|  | OpenbookQA | 57.43% / 85.3% | 78.21% / 100.0% | +20.78 pp |
|  | PiQA | 63.94% / 82.9% | 82.90% / 100.0% | +18.96 pp |
| DeepSeek-R1-Distill-Qwen-1.5B | CommonsenseQA | 45.21% / 94.5% | 45.77% / 96.5% | +0.56 pp |
|  | OpenbookQA | 49.69% / 86.8% | 48.68% / 90.4% | -1.01 pp |
|  | PiQA | 30.75% / 50.0% | 61.35% / 95.3% | +30.60 pp |
| Qwen3-4B-Instruct-2507 | CommonsenseQA | 41.97% / 60.4% | 79.72% / 100.0% | +37.75 pp |
|  | OpenbookQA | 53.56% / 70.7% | 83.71% / 100.0% | +30.15 pp |
|  | PiQA | 68.53% / 95.4% | 87.07% / 100.0% | +18.54 pp |

**Table 1.** Overall accuracy and parse rate for all 24 model-task-prompt runs. Each task has the same item count in both prompt conditions.

The improved prompt achieves a 100% parse rate for Qwen2.5 and Qwen3 on all three tasks. DeepSeek's parse rate also rises on every task, with its largest increase on PiQA (50.0% to 95.3%). In contrast, TinyLlama's parse rate falls substantially under the constrained prompt, and its overall accuracy decreases on all tasks.

## 3. Analysis and Discussion

Qwen3-4B-Instruct-2507 is the strongest model under the improved prompt on every task: 79.72% on CommonsenseQA, 83.71% on OpenbookQA, and 87.07% on PiQA. Its largest gain is on CommonsenseQA (+37.75 pp), where the baseline parse rate was only 60.4%. Qwen2.5-3B-Instruct also improves consistently, gaining 14.65-20.78 pp across tasks while reaching complete parseability.

The gains are not all explained by formatting. Qwen2.5's parsed accuracy (accuracy among parsed outputs) increases from 71.34% to 77.75% on CommonsenseQA, 67.30% to 78.21% on OpenbookQA, and 77.12% to 82.90% on PiQA. Qwen3's parsed accuracy likewise rises from 69.46%, 75.79%, and 71.84% to 79.72%, 83.71%, and 87.07%. This suggests the constrained instruction helped these models both return usable labels and select answers more accurately in this setup.

DeepSeek shows a mixed response. Its parse rate improves across all tasks, but CommonsenseQA accuracy changes only slightly (+0.56 pp) and OpenbookQA accuracy falls by 1.01 pp. On PiQA, overall accuracy rises by 30.60 pp, mostly because many more generations yield a parseable answer; parsed accuracy increases more modestly, from 61.49% to 64.40%. Thus, for this model and task, the main measured benefit is answer completion and format compliance rather than a comparably large improvement among already parsed outputs.

TinyLlama does not reliably follow the instruction to emit a single label. Its improved-prompt parse rates are only 1.7% on CommonsenseQA, 1.2% on OpenbookQA, and 9.1% on PiQA. The prompt modification is therefore not universally effective: an instruction-style constraint that works well with chat-tuned models can fail with a small completion-oriented model. TinyLlama's very small parsed subsets also make its parsed-accuracy figures unstable; overall results and parse rate should be considered together.

## 4. Limitations and Conclusion

The evaluation uses one benchmark split and one deterministic generation per question. Results may differ with another split, decoding strategy, or prompt wording. The parser conservatively maps free-form generations to option labels; ambiguous outputs are left unparsed rather than credited by guessing, and unparsed responses count as incorrect in overall accuracy. The token budgets differ between models to accommodate their output behavior, so cross-model comparisons reflect both model capability and these chosen inference settings. Within each model, however, the baseline and improved conditions use the same budget and decoding configuration.

Overall, a dynamically constrained answer prompt substantially improves accuracy and output parseability for Qwen2.5 and Qwen3, and improves DeepSeek mainly on PiQA. It harms TinyLlama, showing that prompt effectiveness depends on model behavior and template compatibility. Under the improved prompt in this experiment, Qwen3 obtains the highest accuracy on all three tasks.

## References

1. Open-Style, *Open-LLM-Benchmark* dataset. https://huggingface.co/datasets/Open-Style/Open-LLM-Benchmark
2. TinyLlama, *TinyLlama_v1.1*. https://huggingface.co/TinyLlama/TinyLlama_v1.1
3. Qwen, *Qwen2.5-3B-Instruct*. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
4. DeepSeek, *DeepSeek-R1-Distill-Qwen-1.5B*. https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
5. Qwen, *Qwen3-4B-Instruct-2507*. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507