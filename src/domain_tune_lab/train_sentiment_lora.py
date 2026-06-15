from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = "你是中文情感分类器。请严格只输出“正面”或“负面”，不要解释。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on ChnSentiCorp sentiment classification.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", default="data/processed/chnsenticorp_sentiment/train.jsonl")
    parser.add_argument("--valid-file", default="data/processed/chnsenticorp_sentiment/valid.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/qwen2.5-0.5b-chnsenticorp-lora")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def build_prompt(tokenizer: Any, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\n评论：{text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class SentimentSftDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, str]], tokenizer: Any, max_length: int) -> None:
        self.examples: list[dict[str, list[int]]] = []
        eos = tokenizer.eos_token or ""
        for record in records:
            prompt = build_prompt(tokenizer, record["instruction"], record["input"])
            target = record["output"].strip() + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + target, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
            labels = labels[: len(full_ids)]
            if any(label != -100 for label in labels):
                self.examples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "report_to": "none",
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    if "lr_scheduler_type" in signature.parameters:
        kwargs["lr_scheduler_type"] = "cosine"
    if "warmup_ratio" in signature.parameters:
        kwargs["warmup_ratio"] = 0.03
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(Path(args.train_file), args.max_train_samples)
    valid_records = read_jsonl(Path(args.valid_file), args.max_valid_samples)
    train_dataset = SentimentSftDataset(train_records, tokenizer, args.max_length)
    valid_dataset = SentimentSftDataset(valid_records, tokenizer, args.max_length)
    print(f"train examples: {len(train_dataset)}")
    print(f"valid examples: {len(valid_dataset)}")

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
