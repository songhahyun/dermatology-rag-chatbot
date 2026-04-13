from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from medical_extractor import MedicalEntityExtractor
from ollama_client import OllamaClient

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


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
            "질병명": extracted.get("질병명", []),
            "증상": extracted.get("증상", []),
            "처치 방법": extracted.get("처치 방법", []),
            "환자 특성": extracted.get("환자 특성", []),
        },
    }


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_error_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(
    input_path: Path,
    output_path: Path,
    extractor: MedicalEntityExtractor,
    max_records: int | None = None,
    checkpoint_every: int = 50,
) -> None:
    file_start = time.perf_counter()

    with input_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {input_path}")

    if input_path.name == "TL_내과_통합.json":
        data = [record for record in data if record.get("q_type") == 3]
        print(f"[{input_path.name}] filtered by q_type=3 -> {len(data)} records")

    total = len(data) if max_records is None else min(len(data), max_records)
    records = data[:total]

    documents: list[dict[str, Any]] = []
    failed_count = 0

    partial_path = output_path.with_name(f"{output_path.stem}.partial.json")
    errors_path = output_path.with_name(f"{output_path.stem}_errors.jsonl")
    # Start fresh for this run.
    if errors_path.exists():
        errors_path.unlink()

    def save_checkpoint(processed_idx: int) -> None:
        _save_json(partial_path, documents)
        print(
            f"Checkpoint saved: {partial_path} | "
            f"processed={processed_idx}/{total}, success={len(documents)}, failed={failed_count}"
        )

    fatal_error: Exception | None = None

    try:
        for idx, record in enumerate(
            tqdm(records, total=total, desc=f"Processing {input_path.name}", unit="req"),
            start=1,
        ):
            question = str(record.get("question", ""))
            answer = str(record.get("answer", ""))

            req_start = time.perf_counter()
            try:
                extracted = extractor.extract(question=question, answer=answer)
                doc = _to_document(record=record, extracted=extracted)
                documents.append(doc)
            except json.JSONDecodeError as exc:
                failed_count += 1
                _append_error_jsonl(
                    errors_path,
                    {
                        "index": idx,
                        "qa_id": record.get("qa_id"),
                        "error_type": "JSONDecodeError",
                        "error": str(exc),
                        "question": question[:300],
                        "answer": answer[:300],
                    },
                )
            except Exception as exc:
                failed_count += 1
                _append_error_jsonl(
                    errors_path,
                    {
                        "index": idx,
                        "qa_id": record.get("qa_id"),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "question": question[:300],
                        "answer": answer[:300],
                    },
                )

            req_elapsed = time.perf_counter() - req_start
            if idx % 100 == 0 or idx == total:
                print(
                    f"[{input_path.name}] processed {idx}/{total} | "
                    f"time per request: {req_elapsed:.2f}s | "
                    f"success={len(documents)} failed={failed_count}"
                )

            if checkpoint_every > 0 and idx % checkpoint_every == 0:
                save_checkpoint(idx)

    except KeyboardInterrupt as exc:
        fatal_error = exc
    except Exception as exc:
        fatal_error = exc
    finally:
        # Always keep partial output so completed work is not lost.
        save_checkpoint(total if fatal_error is None else len(documents) + failed_count)

    if fatal_error is None:
        _save_json(output_path, documents)
        total_elapsed = time.perf_counter() - file_start
        avg_per_request = total_elapsed / total if total else 0.0
        print(f"Saved final output: {output_path}")
        print(
            f"[{input_path.name}] total elapsed time: {total_elapsed:.2f}s | "
            f"time per request(avg): {avg_per_request:.2f}s | "
            f"success={len(documents)} failed={failed_count}"
        )
    else:
        print(
            f"[{input_path.name}] stopped by fatal error: {fatal_error}. "
            f"Partial output is available at: {partial_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TL/VL 통합 JSON을 Chroma 적재용 Document JSON으로 변환합니다."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/raw/TL_내과_통합.json"],
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
    parser.add_argument("--keep-alive", default="90m", help="Ollama keep_alive 값")
    parser.add_argument("--temperature", type=float, default=0.0, help="생성 온도")
    parser.add_argument("--num-ctx", type=int, default=8192, help="컨텍스트 길이")
    parser.add_argument("--max-records", type=int, default=None, help="샘플 처리 개수 제한")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="중간 저장 주기(레코드 수). 0 이하면 중간 저장 비활성화",
    )
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
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
