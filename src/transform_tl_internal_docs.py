from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Optional import for direct Chroma/LangChain conversion.
try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    Document = None  # type: ignore[assignment]

Q_A_PATTERN = re.compile(r"^\s*Q:\s*(.*?)\s*\nA:\s*(.*)\s*$", re.DOTALL)
PAREN_ALIAS_PATTERN = re.compile(r"\(([^)]*[A-Za-z][^)]*)\)")
ASCII_ALIAS_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b")
DEFAULT_DISEASE_ALIAS_MAP = {
    "천식": ["asthma"],
}


def parse_qa(page_content: str) -> tuple[str, str]:
    m = Q_A_PATTERN.match(page_content or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", (page_content or "").strip()


def infer_primary_disease(question_text: str) -> list[str]:
    text = (question_text or "").strip()
    if not text:
        return []

    # Example: "천식의 급성 악화..." -> "천식"
    m = re.match(r"^([가-힣A-Za-z0-9\-\(\)\s]{1,40}?)의\s", text)
    if not m:
        return []

    candidate = m.group(1).strip()
    if not candidate:
        return []

    # Trim parenthetical English annotation: "과민성 폐장염(Hypersensitivity Pneumonitis)" -> "과민성 폐장염"
    candidate = re.sub(r"\([^)]*\)", "", candidate).strip()

    # Exclude obviously generic starts.
    blocked = ("다음", "아래", "환자", "증상", "원인", "정의", "진단", "치료")
    if candidate.startswith(blocked):
        return []

    return [candidate]


def normalize_terms(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw in values or []:
        s = str(raw).strip()
        if not s:
            continue

        # Pull terms inside parentheses first (e.g., "...(알부테롤) 투여" -> "알부테롤").
        parens = [p.strip() for p in re.findall(r"\(([^)]{1,40})\)", s) if p.strip()]
        candidates = parens if parens else [s]

        for c in candidates:
            c = re.sub(r"\s*(투여|치료|시행|사용)$", "", c).strip()
            c = re.sub(r"\s{2,}", " ", c)
            if c and c not in seen:
                seen.add(c)
                normalized.append(c)

    return normalized


def extract_aliases(disease_names: list[str], *texts: str) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    # Add aliases from curated Korean->English mapping first.
    for disease in disease_names:
        for alias in DEFAULT_DISEASE_ALIAS_MAP.get(disease, []):
            low = alias.lower()
            if low not in seen:
                seen.add(low)
                aliases.append(low)

    for text in texts:
        if not text:
            continue

        # Prefer parenthetical English phrase first.
        for chunk in PAREN_ALIAS_PATTERN.findall(text):
            if not chunk:
                continue
            words = [w for w in ASCII_ALIAS_PATTERN.findall(chunk) if len(w) >= 2]
            if not words:
                continue
            alias = " ".join(words)
            low = alias.lower()
            if low not in seen:
                seen.add(low)
                aliases.append(low)

        # Then single-token aliases (COPD, GERD, asthma, etc.).
        for token in ASCII_ALIAS_PATTERN.findall(text):
            if len(token) < 2:
                continue
            low = token.lower()
            if low not in seen:
                seen.add(low)
                aliases.append(low)

    return aliases


def build_retrieval_text(
    disease_names: list[str],
    symptoms: list[str],
    treatments: list[str],
    patient_features: list[str],
    question_text: str,
    answer_text: str,
) -> str:
    return "\n".join(
        [
            f"질병명: {', '.join(disease_names)}",
            f"증상: {', '.join(symptoms)}",
            f"처치 방법: {', '.join(treatments)}",
            f"환자 특성: {', '.join(patient_features)}",
            f"질문: {question_text}",
            f"답변: {answer_text}",
        ]
    )


def transform_record(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {}) or {}
    qa_id = int(metadata.get("qa_id", 0) or 0)
    source_file = str(metadata.get("source_file", ""))
    source_folder = str(metadata.get("source_folder", ""))
    source_stem = Path(source_file).stem if source_file else str(qa_id)

    question_text, answer_text = parse_qa(str(item.get("page_content", "")))

    disease_names = normalize_terms(metadata.get("질병명", []))
    if not disease_names:
        disease_names = infer_primary_disease(question_text)

    symptoms = normalize_terms(metadata.get("증상", []))
    treatments = normalize_terms(metadata.get("처치 방법", []))
    patient_features = normalize_terms(metadata.get("환자 특성", []))

    aliases = extract_aliases(disease_names, question_text, answer_text, " ".join(disease_names))

    page_content = f"Q: {question_text}\nA: {answer_text}"
    retrieval_text = build_retrieval_text(
        disease_names=disease_names,
        symptoms=symptoms,
        treatments=treatments,
        patient_features=patient_features,
        question_text=question_text,
        answer_text=answer_text,
    )

    return {
        "doc_id": f"{source_folder}:{source_stem}",
        "qa_id": qa_id,
        "source_file": source_file,
        "question_text": question_text,
        "answer_text": answer_text,
        "page_content": page_content,
        "retrieval_text": retrieval_text,
        "disease_names": disease_names,
        "symptoms": symptoms,
        "treatments": treatments,
        "patient_features": patient_features,
        "aliases": aliases,
        "chunk_type": "parent",
    }


def transform_json(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON: {input_path}")

    transformed = [transform_record(item) for item in raw]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    return transformed


def to_chroma_documents(records: list[dict[str, Any]]) -> list[Any]:
    if Document is None:
        raise ImportError(
            "langchain_core 가 설치되어 있지 않습니다. `pip install langchain-core` 후 사용하세요."
        )

    docs: list[Any] = []
    for r in records:
        metadata = {
            "doc_id": r["doc_id"],
            "qa_id": r["qa_id"],
            "source_file": r["source_file"],
            "disease_names": r["disease_names"],
            "symptoms": r["symptoms"],
            "treatments": r["treatments"],
            "patient_features": r["patient_features"],
            "aliases": r["aliases"],
            "chunk_type": r["chunk_type"],
            "retrieval_text": r["retrieval_text"],
        }
        docs.append(Document(page_content=r["page_content"], metadata=metadata, id=r["doc_id"]))
    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform TL_*_documents.json into normalized parent schema.")
    parser.add_argument(
        "--input",
        default="data/processed/chroma_documents/TL_내과_통합_documents.json",
        help="입력 JSON 경로",
    )
    parser.add_argument(
        "--output",
        default="data/processed/chroma_documents/TL_내과_통합_chunks.json",
        help="출력 JSON 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = transform_json(input_path=input_path, output_path=output_path)
    print(f"Saved: {output_path} (records={len(records)})")


if __name__ == "__main__":
    main()
