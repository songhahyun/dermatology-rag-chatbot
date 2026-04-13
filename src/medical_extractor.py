from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ollama_client import OllamaClient


SYSTEM_PROMPT = """당신은 한국어 의학 QA 문장에서 정보 추출만 수행하는 엔진이다.
반드시 JSON 객체 하나만 출력한다.
키는 정확히 ["질병명", "증상", "처치 방법"] 이고, 각 값은 문자열 배열이다.
중복은 제거하고 원문 표현을 최대한 유지한다.
해당 정보가 없으면 빈 배열([])을 넣는다.
설명 문장, 마크다운, 코드블록은 절대 출력하지 않는다."""


FEW_SHOT_EXAMPLES = """[예시 1 입력]
question: 45세 남자가 당뇨와 체중감소로 내원했다. 공복 혈당이 190 mg/dL였다.
answer: 제2형 당뇨병으로 진단 후 식이조절과 메트포르민을 시작한다.
[예시 1 출력]
{"질병명":["제2형 당뇨병"],"증상":["당뇨","체중감소"],"처치 방법":["식이조절","메트포르민 시작"]}

[예시 2 입력]
question: 환자는 흉통을 호소하며 심전도에서 ST 분절 상승이 보였다.
answer: 급성 심근경색으로 판단하여 즉시 관상동맥 중재술을 시행한다.
[예시 2 출력]
{"질병명":["급성 심근경색"],"증상":["흉통","ST 분절 상승"],"처치 방법":["관상동맥 중재술 시행"]}"""


def _extract_first_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"JSON object not found in model output: {text[:300]}")
    return match.group(0)


def _normalize_result(parsed: dict) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key in ("질병명", "증상", "처치 방법"):
        value = parsed.get(key, [])
        if isinstance(value, list):
            cleaned = []
            seen = set()
            for item in value:
                if not isinstance(item, str):
                    continue
                v = item.strip()
                if not v:
                    continue
                if v in seen:
                    continue
                seen.add(v)
                cleaned.append(v)
            normalized[key] = cleaned
        else:
            normalized[key] = []
    return normalized


@dataclass
class MedicalEntityExtractor:
    client: OllamaClient
    model: str
    temperature: float = 0.0
    num_ctx: int = 8192

    def _build_prompt(self, question: str, answer: str) -> str:
        return (
            f"{FEW_SHOT_EXAMPLES}\n\n"
            "[실제 입력]\n"
            f"question: {question}\n"
            f"answer: {answer}\n"
            "[실제 출력]\n"
        )

    def extract(self, question: str, answer: str) -> dict[str, list[str]]:
        prompt = self._build_prompt(question=question, answer=answer)
        raw = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )
        parsed = json.loads(_extract_first_json_object(raw))
        return _normalize_result(parsed)

