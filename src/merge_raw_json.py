from pathlib import Path
import json


project_root = Path(__file__).resolve().parents[1]
raw_root = project_root / "data" / "raw"

# "\ud1b5\ud569" is "통합". Keep it escaped to avoid encoding issues.
MERGED_SUFFIX = "\ud1b5\ud569"
TARGET_PREFIXES = ("VL_", "TL_")


def merge_folder(folder: Path) -> tuple[int, int, Path]:
    """폴더 안의 개별 JSON을 하나의 리스트 JSON으로 합칩니다."""
    output_path = folder.parent / f"{folder.name}_{MERGED_SUFFIX}.json"

    all_records = []
    file_count = 0

    json_files = sorted(folder.glob("*.json"))
    print(f"[{folder.name}] files: {len(json_files)}")

    for file_path in json_files:
        # Skip previously merged output file if it exists in the same folder by any chance.
        if file_path.name.endswith(f"_{MERGED_SUFFIX}.json"):
            continue

        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

        data["source_folder"] = folder.name
        data["source_file"] = file_path.name

        all_records.append(data)
        file_count += 1

    with output_path.open("w", encoding="utf-8-sig") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    return file_count, len(all_records), output_path


def main() -> None:
    """TL/VL 원본 폴더를 찾아 병합 작업을 실행합니다."""
    target_folders = [
        p
        for p in sorted(raw_root.iterdir())
        if p.is_dir() and p.name.startswith(TARGET_PREFIXES)
    ]

    if not target_folders:
        print(f"No target folders found under: {raw_root}")
        return

    total_files = 0
    total_records = 0

    for folder in target_folders:
        file_count, record_count, output_path = merge_folder(folder)
        total_files += file_count
        total_records += record_count

        print(f"Merged files: {file_count}")
        print(f"Total records: {record_count}")
        print(f"Saved to: {output_path}\n")

    print("=== Summary ===")
    print(f"Folders processed: {len(target_folders)}")
    print(f"Total merged files: {total_files}")
    print(f"Total merged records: {total_records}")


if __name__ == "__main__":
    main()
