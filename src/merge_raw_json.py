import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

MODE_CONFIG = {
    "tlvl": {
        "prefixes": ("VL_내과", "TL_내과"),
        "suffix": "통합",
        "add_source_fields": True,
    },
    "ts": {
        "prefixes": ("TS_",),
        "suffix": "merged",
        "add_source_fields": False,
    },
}


def merge_folder(
    folder: Path,
    output_suffix: str,
    add_source_fields: bool,
    write_encoding: str = "utf-8",
) -> tuple[int, int, Path]:
    output_path = folder.parent / f"{folder.name}_{output_suffix}.json"
    json_files = sorted(folder.glob("*.json"), key=lambda p: p.name)

    all_records = []
    file_count = 0
    for file_path in json_files:
        if file_path.name.endswith(f"_{output_suffix}.json") or file_path.name == "merged.json":
            continue

        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if add_source_fields and isinstance(data, dict):
            data["source_folder"] = folder.name
            data["source_file"] = file_path.name

        all_records.append(data)
        file_count += 1

    with output_path.open("w", encoding=write_encoding) as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    return file_count, len(all_records), output_path


def merge_by_mode(raw_root: Path, mode: str) -> None:
    config = MODE_CONFIG[mode]
    prefixes = config["prefixes"]
    suffix = config["suffix"]
    add_source_fields = config["add_source_fields"]

    target_folders = [
        p
        for p in sorted(raw_root.iterdir(), key=lambda p: p.name)
        if p.is_dir() and p.name.startswith(prefixes)
    ]

    if not target_folders:
        print(f"No target folders found under: {raw_root} for mode: {mode}")
        return

    total_files = 0
    total_records = 0

    for folder in target_folders:
        file_count, record_count, output_path = merge_folder(folder, suffix, add_source_fields)
        total_files += file_count
        total_records += record_count

        print(f"[{folder.name}] merged files: {file_count}")
        print(f"Saved to: {output_path}\n")

    print("=== Summary ===")
    print(f"Mode: {mode}")
    print(f"Folders processed: {len(target_folders)}")
    print(f"Total merged files: {total_files}")
    print(f"Total merged records: {total_records}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSON files under data/raw by mode.")
    parser.add_argument(
        "--mode",
        choices=("tlvl", "ts"),
        default="tlvl",
        help="tlvl: VL_내과/TL_내과 -> *_통합.json, ts: TS_* -> *_merged.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_ROOT,
        help=f"Root raw directory (default: {RAW_ROOT}).",
    )
    args = parser.parse_args()

    merge_by_mode(args.raw_dir, args.mode)


if __name__ == "__main__":
    main()
    # python src/merge_raw_json.py --mode tlvl
    # python src/merge_raw_json.py --mode ts
