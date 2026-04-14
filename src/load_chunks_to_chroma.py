from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def to_scalar_metadata(record: dict[str, Any]) -> dict[str, Any]:
    # Chroma metadata values must be scalar (str/int/float/bool).
    disease_names = record.get("disease_names", []) or []
    symptoms = record.get("symptoms", []) or []
    treatments = record.get("treatments", []) or []
    patient_features = record.get("patient_features", []) or []
    aliases = record.get("aliases", []) or []

    metadata: dict[str, Any] = {
        "doc_id": str(record.get("doc_id", "")),
        "qa_id": int(record.get("qa_id", 0) or 0),
        "source_file": str(record.get("source_file", "")),
        "question_text": str(record.get("question_text", "")),
        "answer_text": str(record.get("answer_text", "")),
        "retrieval_text": str(record.get("retrieval_text", "")),
        "chunk_type": str(record.get("chunk_type", "parent")),
        "disease_names": json.dumps(disease_names, ensure_ascii=False),
        "symptoms": json.dumps(symptoms, ensure_ascii=False),
        "treatments": json.dumps(treatments, ensure_ascii=False),
        "patient_features": json.dumps(patient_features, ensure_ascii=False),
        "aliases": json.dumps(aliases, ensure_ascii=False),
    }
    return metadata


def batched(iterable: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [iterable[i : i + batch_size] for i in range(0, len(iterable), batch_size)]


def ingest_chunks_to_chroma(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str,
    batch_size: int = 200,
    use_retrieval_text: bool = True,
    reset_collection: bool = False,
) -> None:
    rows = load_chunks(chunks_path)
    if not rows:
        raise ValueError(f"No records in input: {chunks_path}")

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if reset_collection:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)

    seen_ids: set[str] = set()
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError("Found empty doc_id")
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate doc_id found: {doc_id}")
        seen_ids.add(doc_id)

    total = len(rows)
    done = 0

    for chunk in batched(rows, batch_size=batch_size):
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for row in chunk:
            doc_id = str(row["doc_id"])
            text_field = "retrieval_text" if use_retrieval_text else "page_content"
            doc_text = str(row.get(text_field, "") or row.get("page_content", ""))

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(to_scalar_metadata(row))

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        done += len(ids)
        print(f"Upserted {done}/{total}")

    print(f"Done. Persist dir: {persist_dir}")
    print(f"Collection: {collection_name}, count={collection.count()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load *_chunks.json into Chroma PersistentClient")
    parser.add_argument(
        "--input",
        default="data/processed/chroma_documents/TL_내과_통합_chunks.json",
        help="청크 JSON 경로",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/processed/chroma_db/TL_내과_통합",
        help="Chroma persist 디렉토리",
    )
    parser.add_argument(
        "--collection-name",
        default="TL_내과_통합",
        help="Chroma collection 이름",
    )
    parser.add_argument("--batch-size", type=int, default=200, help="배치 크기")
    parser.add_argument(
        "--text-field",
        choices=["retrieval_text", "page_content"],
        default="retrieval_text",
        help="임베딩 대상 텍스트 필드",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="기존 collection 삭제 후 재생성",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_chunks_to_chroma(
        chunks_path=Path(args.input),
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        use_retrieval_text=(args.text_field == "retrieval_text"),
        reset_collection=args.reset_collection,
    )


if __name__ == "__main__":
    main()
