import json
from typing import Any, Mapping, Sequence
from urllib import error, request

from mera_openjudge.backends.base import BaseJudgeBackend
from mera_openjudge.exceptions import JudgeBackendError


class OpenAICompatibleJudgeBackend(BaseJudgeBackend):
    def __init__(self, config: Mapping[str, Any]):
        self.base_url = str(config.get("base_url", "")).strip()
        self.api_key = str(config.get("api_key", "")).strip()
        self.model_name = str(config.get("model_name", "")).strip()
        self.temperature = float(config.get("temperature", 0.1))
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.timeout = int(config.get("timeout", 120))
        if not self.base_url:
            raise JudgeBackendError("OpenAI-compatible judge backend requires base_url.")
        if not self.model_name:
            raise JudgeBackendError("OpenAI-compatible judge backend requires model_name.")

    def _build_endpoint(self):
        normalized_url = self.base_url.rstrip("/")
        if normalized_url.endswith("/chat/completions"):
            return normalized_url
        if normalized_url.endswith("/v1"):
            return normalized_url + "/chat/completions"
        return normalized_url + "/v1/chat/completions"

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompts: Sequence[str]):
        endpoint = self._build_endpoint()
        headers = self._build_headers()
        outputs = []
        for prompt in prompts:
            payload = json.dumps(
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_new_tokens,
                    "n": 1,
                }
            ).encode("utf-8")
            req = request.Request(
                endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="ignore")
                raise JudgeBackendError(
                    f"OpenAI-compatible judge backend HTTP {exc.code}: {details}"
                ) from exc
            except error.URLError as exc:
                raise JudgeBackendError(
                    f"OpenAI-compatible judge backend connection failed: {exc.reason}"
                ) from exc

            try:
                data = json.loads(body)
                choice = data["choices"][0]
                if "message" in choice:
                    content = choice["message"]["content"]
                else:
                    content = choice["text"]
                outputs.append(str(content).strip())
            except Exception as exc:
                raise JudgeBackendError(
                    "OpenAI-compatible judge backend returned an invalid response."
                ) from exc
        return outputs
