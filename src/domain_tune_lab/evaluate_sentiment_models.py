from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = "你是中文情感分类器。请严格只输出“正面”或“负面”，不要解释。"
LABELS = ("正面", "负面")


@dataclass
class Prediction:
    index: int
    input: str
    expected: str
    raw_output: str
    predicted: str | None
    latency_sec: float

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected

    @property
    def valid_format(self) -> bool:
        return self.predicted in LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Base vs LoRA on ChnSentiCorp sentiment classification.")
    parser.add_argument("--test-file", default="data/processed/chnsenticorp_sentiment/test.jsonl")
    parser.add_argument("--output-dir", default="outputs/chnsenticorp_eval")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", default="checkpoints/qwen2.5-0.5b-chnsenticorp-lora")
    parser.add_argument("--modes", default="base,lora", help="Comma-separated: base,lora")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, max_samples: int = 0) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples > 0 and len(records) >= max_samples:
                break
    return records


def build_prompt(tokenizer: Any, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\n评论：{text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_label(output: str) -> str | None:
    cleaned = output.strip().replace("：", ":")
    if cleaned.startswith("正面"):
        return "正面"
    if cleaned.startswith("负面"):
        return "负面"
    has_pos = "正面" in cleaned or "积极" in cleaned or "好评" in cleaned
    has_neg = "负面" in cleaned or "消极" in cleaned or "差评" in cleaned
    if has_pos and not has_neg:
        return "正面"
    if has_neg and not has_pos:
        return "负面"
    return None


def make_quant_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model(base_model: str, adapter_path: str | None, load_in_4bit: bool) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    quant_config = make_quant_config(load_in_4bit)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(tokenizer: Any, model: Any, instruction: str, text: str, max_new_tokens: int) -> tuple[str, float]:
    prompt = build_prompt(tokenizer, instruction, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency = time.perf_counter() - started
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), latency


def compute_metrics(predictions: list[Prediction]) -> dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {}
    valid = [item for item in predictions if item.valid_format]
    correct = sum(1 for item in predictions if item.correct)
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = sum(1 for item in predictions if item.expected == label and item.predicted == label)
        fp = sum(1 for item in predictions if item.expected != label and item.predicted == label)
        fn = sum(1 for item in predictions if item.expected == label and item.predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "total": total,
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "format_valid_rate": len(valid) / total,
        "avg_latency_sec": sum(item.latency_sec for item in predictions) / total,
        "per_label": per_label,
        "confusion": {
            "pos_as_pos": sum(1 for item in predictions if item.expected == "正面" and item.predicted == "正面"),
            "pos_as_neg": sum(1 for item in predictions if item.expected == "正面" and item.predicted == "负面"),
            "pos_invalid": sum(1 for item in predictions if item.expected == "正面" and item.predicted is None),
            "neg_as_neg": sum(1 for item in predictions if item.expected == "负面" and item.predicted == "负面"),
            "neg_as_pos": sum(1 for item in predictions if item.expected == "负面" and item.predicted == "正面"),
            "neg_invalid": sum(1 for item in predictions if item.expected == "负面" and item.predicted is None),
        },
    }


def write_jsonl(path: Path, predictions: list[Prediction]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in predictions:
            file.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")


def write_report(path: Path, metrics_by_mode: dict[str, dict[str, Any]], predictions_by_mode: dict[str, list[Prediction]]) -> None:
    lines = [
        "# ChnSentiCorp Base vs LoRA 评测报告",
        "",
        "评测任务：公开中文评论二分类，只允许输出 `正面` 或 `负面`。",
        "",
        "| 模型 | Accuracy | Macro F1 | Format Valid | Avg Latency |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, metrics in metrics_by_mode.items():
        lines.append(
            f"| {mode} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.2%} | "
            f"{metrics['format_valid_rate']:.2%} | {metrics['avg_latency_sec']:.2f}s |"
        )
    lines.extend(["", "## 混淆矩阵", ""])
    for mode, metrics in metrics_by_mode.items():
        lines.append(f"### {mode}")
        for key, value in metrics["confusion"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(["## 典型错误样例", ""])
    for mode, predictions in predictions_by_mode.items():
        lines.append(f"### {mode}")
        failures = [item for item in predictions if not item.correct][:8]
        if not failures:
            lines.append("- 无错误样例。")
        for item in failures:
            raw = item.raw_output.replace("\n", " ")[:120]
            text = item.input.replace("\n", " ")[:120]
            lines.append(f"- 期望 `{item.expected}`，预测 `{item.predicted}`，输出：{raw}；评论：{text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_mode(
    mode: str,
    records: list[dict[str, str]],
    base_model: str,
    adapter_path: str,
    load_in_4bit: bool,
    max_new_tokens: int,
) -> list[Prediction]:
    adapter = None if mode == "base" else adapter_path
    if adapter is not None and not Path(adapter).exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
    tokenizer, model = load_model(base_model, adapter, load_in_4bit)
    predictions: list[Prediction] = []
    for index, record in enumerate(records):
        raw, latency = generate_one(tokenizer, model, record["instruction"], record["input"], max_new_tokens)
        predictions.append(
            Prediction(
                index=index,
                input=record["input"],
                expected=record["output"],
                raw_output=raw,
                predicted=extract_label(raw),
                latency_sec=latency,
            )
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.test_file), args.max_samples)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    metrics_by_mode: dict[str, dict[str, Any]] = {}
    predictions_by_mode: dict[str, list[Prediction]] = {}
    for mode in modes:
        if mode not in {"base", "lora"}:
            raise ValueError(f"Unsupported mode: {mode}")
        predictions = evaluate_mode(
            mode=mode,
            records=records,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
        )
        predictions_by_mode[mode] = predictions
        metrics_by_mode[mode] = compute_metrics(predictions)
        write_jsonl(output_dir / f"eval_{mode}.jsonl", predictions)

    (output_dir / "metrics.json").write_text(json.dumps(metrics_by_mode, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "eval_report.md", metrics_by_mode, predictions_by_mode)
    print(json.dumps(metrics_by_mode, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
