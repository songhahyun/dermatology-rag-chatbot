from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from medical_extractor import MedicalEntityExtractor
from ollama_client import OllamaClient


def _to_document(record: dict[str, Any], extracted: dict[str, list[str]]) -> dict[str, Any]:
    qa_id = record.get("qa_id", "unknown")
    source_folder = record.get("source_folder", "unknown")
    source_file = record.get("source_file", "unknown")

    return {
        "id": f"{source_folder}:{source_file}:{qa_id}",
        "page_content": f"Q: {record.get('question', '')}\nA: {record.get('answer', '')}",
        "metadata": {
            "qa_id": qa_id,
            "domain": record.get("domain"),
            "q_type": record.get("q_type"),
            "source_folder": source_folder,
            "source_file": source_file,
            "질병명": extracted["질병명"],
            "증상": extracted["증상"],
            "처치 방법": extracted["처치 방법"],
            "환자 특성": extracted["환자 특성"],
        },
    }


def process_file(
    input_path: Path,
    output_path: Path,
    extractor: MedicalEntityExtractor,
    max_records: int | None = None,
) -> None:
    file_start = time.perf_counter()
    with input_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {input_path}")

    documents: list[dict[str, Any]] = []
    total = len(data) if max_records is None else min(len(data), max_records)

    for idx, record in enumerate(data[:total], start=1):
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))

        req_start = time.perf_counter()
        extracted = extractor.extract(question=question, answer=answer)
        req_elapsed = time.perf_counter() - req_start

        doc = _to_document(record=record, extracted=extracted)
        documents.append(doc)

        if idx % 100 == 0 or idx == total:
            print(
                f"[{input_path.name}] processed {idx}/{total} | "
                f"time per request: {req_elapsed:.2f}s"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - file_start
    avg_per_request = total_elapsed / total if total else 0.0
    print(f"Saved: {output_path}")
    print(
        f"[{input_path.name}] total elapsed time: {total_elapsed:.2f}s | "
        f"time per request(avg): {avg_per_request:.2f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TL/VL 통합 JSON을 Chroma 적재용 Document JSON으로 변환합니다."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/raw/TL_내과_통합.json",
            "data/raw/VL_내과_통합.json",
        ],
        help="입력 JSON 파일 목록",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/chroma_documents",
        help="출력 디렉터리",
    )
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama 모델명")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="HTTP timeout(초)")
    parser.add_argument("--keep-alive", default="30m", help="Ollama keep_alive 값")
    parser.add_argument("--temperature", type=float, default=0.0, help="생성 온도")
    parser.add_argument("--num-ctx", type=int, default=8192, help="컨텍스트 길이")
    parser.add_argument("--max-records", type=int, default=None, help="샘플 처리 개수 제한")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OllamaClient(
        base_url=args.ollama_url,
        timeout_seconds=args.timeout_seconds,
        keep_alive=args.keep_alive,
    )
    extractor = MedicalEntityExtractor(
        client=client,
        model=args.model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
    )

    print(f"Warming up model: {args.model} (keep_alive={args.keep_alive})")
    client.warmup(model=args.model, num_ctx=args.num_ctx)

    output_dir = Path(args.output_dir)
    for input_str in args.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"Skip missing file: {input_path}")
            continue

        output_path = output_dir / f"{input_path.stem}_documents.json"
        process_file(
            input_path=input_path,
            output_path=output_path,
            extractor=extractor,
            max_records=args.max_records,
        )


if __name__ == "__main__":
    main()
