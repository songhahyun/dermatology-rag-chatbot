from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ollama_client import OllamaClient


SYSTEM_PROMPT = """당신은 한국어 의학 QA 문장에서 정보 추출만 수행하는 엔진이다.
반드시 JSON 객체 하나만 출력한다.
키는 정확히 [\"질병명\", \"증상\", \"처치 방법\"] 이고, 각 값은 문자열 배열이다.
질병명은 질환/증후군/진단명만 포함한다.
증상은 환자 호소, 징후, 검사 이상 소견을 포함한다.
처치 방법은 검사, 시술, 약물, 치료, 수술, 생활교정 등 의료 행위를 포함한다.
객관식 번호(예: \"4)\")는 제거한다.
중복은 제거하고 원문 표현을 최대한 유지한다.
해당 정보가 없으면 빈 배열([])을 넣는다.
설명 문장, 마크다운, 코드블록은 절대 출력하지 않는다."""


FALLBACK_SYSTEM_PROMPT = """당신은 한국어 의학 QA 추출기다.
1차 추출이 모두 빈 배열일 때만 호출된다.
질병명이 불명확하면 빈 배열 유지 가능하지만, 증상과 처치 방법은 question/answer에서 최대한 유의미하게 뽑아라.
추측성 진단명은 넣지 않는다. 명시 근거가 없는 질병명은 빈 배열로 둔다.
반드시 JSON 객체 하나만 출력하고 키는 [\"질병명\", \"증상\", \"처치 방법\"] 만 사용한다."""


FEW_SHOT_EXAMPLES = """[예시 1 입력]
question: 45세 남자가 다뇨와 다음, 체중감소로 내원했다. 공복 혈당은 190 mg/dL였다.
answer: 제2형 당뇨병으로 진단 후 식이조절과 메트포르민을 시작한다.
[예시 1 출력]
{"질병명":["제2형 당뇨병"],"증상":["다뇨","다음","체중감소","공복 혈당 190 mg/dL"],"처치 방법":["식이조절","메트포르민 시작"]}

[예시 2 입력]
question: 환자는 흉통을 호소하며 심전도에서 ST 분절 상승이 보였다.
answer: 급성 심근경색으로 판단하여 즉시 관상동맥 중재술을 시행한다.
[예시 2 출력]
{"질병명":["급성 심근경색"],"증상":["흉통","ST 분절 상승"],"처치 방법":["관상동맥 중재술 시행"]}

[예시 3 입력]
question: 25세 남자가 피로, 쉽게 멍이 듦, 운동 시 호흡곤란을 호소한다. 체온은 38.3°C이다.
answer: 4) 골수흡인 및 생검
[예시 3 출력]
{"질병명":[],"증상":["피로","쉽게 멍이 듦","운동 시 호흡곤란","발열(38.3°C)"],"처치 방법":["골수흡인 및 생검"]}

[예시 4 입력]
question: 기침이 3개월 지속되고 밤에 누우면 심해진다. 흉부 X선은 정상이다.
answer: 4) 메타콜린 기관지유발검사
[예시 4 출력]
{"질병명":[],"증상":["만성 기침","야간 악화 기침","흉부 X선 정상"],"처치 방법":["메타콜린 기관지유발검사"]}

[예시 5 입력]
question: 손발 저림과 보행장애가 있고 비타민 B12가 낮다.
answer: 비타민 B12 결핍에 의한 아급성 연합변성으로 진단하고 비타민 B12를 근육주사한다.
[예시 5 출력]
{"질병명":["아급성 연합변성","비타민 B12 결핍"],"증상":["손발 저림","보행장애","비타민 B12 감소"],"처치 방법":["비타민 B12 근육주사"]}"""


def _extract_first_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"JSON object not found in model output: {text[:300]}")
    return match.group(0)


def _cleanup_term(value: str) -> str:
    value = re.sub(r"^\s*\d+\)\s*", "", value)
    value = re.sub(r"^\s*[-*]\s*", "", value)
    return value.strip()


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
                v = _cleanup_term(item)
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


def _all_empty(result: dict[str, list[str]]) -> bool:
    return not (result["질병명"] or result["증상"] or result["처치 방법"])


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

    def _build_fallback_prompt(self, question: str, answer: str) -> str:
        return (
            "[실제 입력]\n"
            f"question: {question}\n"
            f"answer: {answer}\n\n"
            "[요구사항]\n"
            "- question/answer에서 증상과 처치 방법을 최대한 추출\n"
            "- 질병명은 명시된 경우만 추출\n"
            "- JSON만 출력\n"
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
        result = _normalize_result(parsed)

        if _all_empty(result) and (question.strip() or answer.strip()):
            fallback_raw = self.client.generate(
                model=self.model,
                prompt=self._build_fallback_prompt(question=question, answer=answer),
                system=FALLBACK_SYSTEM_PROMPT,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
            )
            fallback_parsed = json.loads(_extract_first_json_object(fallback_raw))
            result = _normalize_result(fallback_parsed)

        return result
