from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mera_openjudge.backends.base import BaseJudgeBackend
from mera_openjudge.exceptions import JudgeBackendError


class HFJudgeBackend(BaseJudgeBackend):
    _RESOURCE_CACHE = {}

    def __init__(self, config: Mapping[str, Any]):
        self.model_name = str(config.get("model_name", "")).strip()
        if not self.model_name:
            raise JudgeBackendError("HF judge backend requires model_name.")
        self.device = str(config.get("device", "cuda")).strip() or "cuda"
        self.batch_size = int(config.get("batch_size", 8))
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.temperature = float(config.get("temperature", 0.1))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.max_input_length = config.get("max_input_length")
        self.model, self.tokenizer = self._get_resources()

    def _get_resources(self):
        cache_key = (self.model_name, self.device, self.trust_remote_code)
        if cache_key not in self._RESOURCE_CACHE:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )
                model.to(self.device)
                model.eval()
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                self._RESOURCE_CACHE[cache_key] = (model, tokenizer)
            except Exception as exc:
                raise JudgeBackendError(
                    f"Failed to load HF judge backend model '{self.model_name}'."
                ) from exc
        return self._RESOURCE_CACHE[cache_key]

    def _render_chat_prompts(self, prompts: Sequence[str]):
        rendered_prompts = []
        for prompt in prompts:
            if getattr(self.tokenizer, "chat_template", None):
                rendered_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered_prompt = prompt
            rendered_prompts.append(rendered_prompt)
        return rendered_prompts

    def generate(self, prompts: Sequence[str]):
        if not prompts:
            return []

        rendered_prompts = self._render_chat_prompts(prompts)
        outputs = []
        try:
            for batch_start in range(0, len(rendered_prompts), self.batch_size):
                batch = rendered_prompts[batch_start : batch_start + self.batch_size]
                tokenizer_kwargs = {
                    "return_tensors": "pt",
                    "padding": True,
                }
                if self.max_input_length is not None:
                    tokenizer_kwargs.update(
                        {
                            "truncation": True,
                            "max_length": int(self.max_input_length),
                        }
                    )
                inputs = self.tokenizer(batch, **tokenizer_kwargs).to(self.device)
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if self.temperature > 0:
                    generation_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": self.temperature,
                        }
                    )
                else:
                    generation_kwargs["do_sample"] = False

                with torch.no_grad():
                    sequences = self.model.generate(**inputs, **generation_kwargs)

                prompt_token_count = inputs["input_ids"].shape[1]
                for sequence in sequences:
                    generated_ids = sequence[prompt_token_count:]
                    outputs.append(
                        self.tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True,
                        ).strip()
                    )
        except Exception as exc:
            raise JudgeBackendError("HF judge backend generation failed.") from exc
        return outputs
