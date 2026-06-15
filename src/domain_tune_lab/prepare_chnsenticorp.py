from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


INSTRUCTION = "判断下面中文评论的情感倾向，只输出“正面”或“负面”。"
DEFAULT_DATASETS = ("lansinuote/ChnSentiCorp", "seamew/ChnSentiCorp")
TEXT_COLUMNS = ("text", "review", "sentence", "comment", "content")
LABEL_COLUMNS = ("label", "labels", "sentiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public ChnSentiCorp as Alpaca JSONL.")
    parser.add_argument("--dataset", default="", help="Hugging Face dataset id. Empty means auto fallback.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/chnsenticorp_sentiment",
        help="Output directory for train/valid/test JSONL files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap for quick experiments.")
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    return parser.parse_args()


def load_public_chnsenticorp(dataset_id: str) -> DatasetDict:
    candidates = (dataset_id,) if dataset_id else DEFAULT_DATASETS
    errors: list[str] = []
    for candidate in candidates:
        try:
            loaded = load_dataset(candidate)
            if not isinstance(loaded, DatasetDict):
                raise TypeError(f"{candidate} did not return a DatasetDict")
            return loaded
        except Exception as exc:  # pragma: no cover - depends on network/source availability
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Failed to load ChnSentiCorp from public sources:\n" + "\n".join(errors))


def detect_column(sample: dict[str, Any], candidates: Iterable[str], role: str) -> str:
    for name in candidates:
        if name in sample:
            return name
    raise KeyError(f"Cannot find {role} column in sample keys: {sorted(sample)}")


def normalize_label(value: Any, features: Any | None = None) -> str:
    if hasattr(features, "int2str") and isinstance(value, int):
        value = features.int2str(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "positive", "pos", "正面", "好评"}:
            return "正面"
        if lowered in {"0", "negative", "neg", "负面", "差评"}:
            return "负面"
    if isinstance(value, bool):
        return "正面" if value else "负面"
    if isinstance(value, int):
        if value == 1:
            return "正面"
        if value == 0:
            return "负面"
    raise ValueError(f"Unsupported sentiment label: {value!r}")


def ensure_splits(dataset: DatasetDict, seed: int) -> DatasetDict:
    if {"train", "validation", "test"}.issubset(dataset.keys()):
        return dataset
    if {"train", "valid", "test"}.issubset(dataset.keys()):
        return DatasetDict(
            train=dataset["train"],
            validation=dataset["valid"],
            test=dataset["test"],
        )
    if {"train", "test"}.issubset(dataset.keys()):
        split = dataset["train"].train_test_split(test_size=0.1, seed=seed)
        return DatasetDict(train=split["train"], validation=split["test"], test=dataset["test"])
    if "train" in dataset:
        first = dataset["train"].train_test_split(test_size=0.2, seed=seed)
        second = first["test"].train_test_split(test_size=0.5, seed=seed)
        return DatasetDict(train=first["train"], validation=second["train"], test=second["test"])
    raise KeyError(f"Unsupported dataset splits: {list(dataset.keys())}")


def to_alpaca_records(split: Dataset) -> list[dict[str, str]]:
    if len(split) == 0:
        return []
    sample = dict(split[0])
    text_col = detect_column(sample, TEXT_COLUMNS, "text")
    label_col = detect_column(sample, LABEL_COLUMNS, "label")
    label_features = split.features.get(label_col)
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in split:
        text = str(row[text_col]).strip()
        if not text:
            continue
        output = normalize_label(row[label_col], label_features)
        key = (text, output)
        if key in seen:
            continue
        seen.add(key)
        records.append({"instruction": INSTRUCTION, "input": text, "output": output})
    return records


def maybe_cap(records: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or len(records) <= limit:
        return records
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    dataset = ensure_splits(load_public_chnsenticorp(args.dataset), args.seed)
    output_dir = Path(args.output_dir)

    split_specs = {
        "train": (dataset["train"], args.max_train),
        "valid": (dataset["validation"], args.max_valid),
        "test": (dataset["test"], args.max_test),
    }
    stats: dict[str, Any] = {"instruction": INSTRUCTION, "splits": {}}
    for name, (split, limit) in split_specs.items():
        records = maybe_cap(to_alpaca_records(split), limit, args.seed)
        write_jsonl(output_dir / f"{name}.jsonl", records)
        stats["splits"][name] = {
            "records": len(records),
            "positive": sum(1 for item in records if item["output"] == "正面"),
            "negative": sum(1 for item in records if item["output"] == "负面"),
        }

    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
