from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DATASET_INFO = {
    "chnsenticorp_sentiment": {
        "file_name": "chnsenticorp_sentiment/train.jsonl",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy prepared data and register it in LLaMA-Factory.")
    parser.add_argument("--project-root", default=".", help="domain-tune-lab root.")
    parser.add_argument("--llamafactory-dir", default="/mnt/d/home/work/tools/LLaMA-Factory")
    parser.add_argument("--dataset-name", default="chnsenticorp_sentiment")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    llama_dir = Path(args.llamafactory_dir).resolve()
    dataset_name = args.dataset_name

    source = project_root / "data" / "processed" / dataset_name
    if not source.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {source}")
    target = llama_dir / "data" / dataset_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

    dataset_info_path = llama_dir / "data" / "dataset_info.json"
    dataset_info = read_json(dataset_info_path)
    dataset_info.update(DATASET_INFO)
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    local_info_path = project_root / "configs" / "llamafactory_dataset_info.json"
    local_info = read_json(local_info_path)
    local_info.update(DATASET_INFO)
    local_info_path.write_text(json.dumps(local_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Synced {source} -> {target}")
    print(f"Registered {dataset_name} in {dataset_info_path}")


if __name__ == "__main__":
    main()
