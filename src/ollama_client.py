from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 300
    keep_alive: str = "30m"

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Rough token estimate for mixed Korean/English text."""
        if not text:
            return 0
        by_chars = len(text) // 2
        by_bytes = len(text.encode("utf-8")) // 3
        return max(by_chars, by_bytes)

    def _validate_num_ctx(self, prompt: str, system: str, num_ctx: int) -> None:
        estimated_prompt_tokens = self._estimate_token_count(prompt)
        estimated_system_tokens = self._estimate_token_count(system)
        reserved_tokens = 512
        needed = estimated_prompt_tokens + estimated_system_tokens + reserved_tokens
        if needed > num_ctx:
            raise ValueError(
                f"num_ctx={num_ctx} is likely too small (estimated needed >= {needed}). "
                "Increase num_ctx or shorten input text."
            )

    def _normalized_base_url(self) -> str:
        """Normalize user input like http://host:11434/api -> http://host:11434."""
        url = self.base_url.rstrip("/")
        if url.endswith("/api"):
            url = url[:-4]
        return url

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
    ) -> str:
        json_guard = (
            "Return ONLY one valid JSON value. "
            "Do not include markdown, code fences, comments, or explanations."
        )
        effective_system = f"{system}\n\n{json_guard}" if system else json_guard
        effective_prompt = (
            f"{prompt}\n\n"
            "[Output Rule]\n"
            "Output must be strict JSON only."
        )

        self._validate_num_ctx(
            prompt=effective_prompt,
            system=effective_system,
            num_ctx=num_ctx,
        )

        base_url = self._normalized_base_url()
        url = f"{base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": effective_prompt,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "system": effective_system,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    "Ollama request failed: HTTP 404 Not Found. "
                    f"Request URL={url}. "
                    "Check that Ollama server is running and use base URL like "
                    "'http://localhost:11434' (without /api or /v1)."
                ) from exc
            raise RuntimeError(f"Ollama request failed: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = json.loads(body)
        if "response" not in parsed:
            raise RuntimeError(f"Unexpected Ollama response: {body[:300]}")

        response_text = parsed["response"].strip()
        obj = json.loads(response_text)
        return json.dumps(obj, ensure_ascii=False)

    def warmup(self, model: str, num_ctx: int = 8192) -> None:
        """Preload model with keep_alive so later requests are faster."""
        _ = self.generate(
            model=model,
            prompt='{"질병명":[],"증상":[],"처치 방법":[]}',
            system="Output exactly the same JSON.",
            temperature=0.0,
            num_ctx=num_ctx,
        )
