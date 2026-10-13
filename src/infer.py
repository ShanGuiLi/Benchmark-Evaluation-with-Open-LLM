"""Model loading and inference wrapper.

Handles both plain completion models and instruction-tuned models that ship a
chat template. When a chat template is available we use it so the model sees a
well-formed conversation; otherwise we feed the raw prompt.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    """A thin wrapper around a HF causal LM for deterministic-ish generation."""

    def __init__(
        self,
        hf_id: str,
        use_chat_template: bool = False,
        device_map: str = "auto",
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> None:
        self.hf_id = hf_id
        self.use_chat_template = use_chat_template
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

        # Default to the best dtype for this hardware. Leaving it as None would
        # make from_pretrained fall back to float32 and blow up GPU memory.
        if dtype is None:
            dtype = pick_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map=device_map,
            dtype=dtype,
        )
        self.model.eval()
        self.device = self.model.device

        # We always drive output length with max_new_tokens. Clearing the
        # inherited max_length avoids transformers warning on every single call
        # that both max_new_tokens and max_length are set.
        try:
            self.model.generation_config.max_length = None
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, prompt: str):
        """Tokenise ``prompt``, using the model's chat template when enabled.

        Instruct models are trained on a specific dialogue format; feeding them
        a raw completion prompt degrades instruction-following. When the model
        ships a chat template (and ``use_chat_template`` is set) we wrap the
        prompt as a single user turn and let the template add the generation
        cue. Base/completion models fall back to raw tokenisation.
        """
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            templated = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # Depending on the transformers version this is either a tensor or
            # a BatchEncoding; normalise to (input_ids, attention_mask).
            if hasattr(templated, "input_ids"):
                input_ids = templated.input_ids
                attention_mask = getattr(templated, "attention_mask", None)
            else:
                input_ids = templated
                attention_mask = None
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask.to(self.device)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Return only the newly generated text for ``prompt``."""
        input_ids, attention_mask = self._prepare_inputs(prompt)
        prompt_len = input_ids.shape[-1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # temperature is only meaningful for sampling; passing it to a greedy
        # decode triggers a warning and is ignored anyway.
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        outputs = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        # Decode only the generated continuation (skip the prompt tokens).
        generated = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def pick_dtype(prefer_bf16: bool = True) -> torch.dtype:
    """Choose a sensible default dtype for the available hardware."""
    if torch.cuda.is_available():
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
