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

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["system"] = system

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = json.loads(body)
        if "response" not in parsed:
            raise RuntimeError(f"Unexpected Ollama response: {body[:300]}")

        return parsed["response"].strip()

    def warmup(self, model: str, num_ctx: int = 8192) -> None:
        """모델을 미리 로드하고 keep_alive를 적용합니다."""
        _ = self.generate(
            model=model,
            prompt='{"질병명":[],"증상":[],"처치 방법":[]}',
            system="반드시 입력을 그대로 출력하라.",
            temperature=0.0,
            num_ctx=num_ctx,
        )
