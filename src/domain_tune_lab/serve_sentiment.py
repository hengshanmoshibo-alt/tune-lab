from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from domain_tune_lab.prepare_chnsenticorp import INSTRUCTION


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/qwen2.5-0.5b-chnsenticorp-lora"
STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend"
METRICS_PATH = Path("outputs/chnsenticorp_eval/metrics.json")
SYSTEM_PROMPT = "你是中文情感分类器。请严格只输出“正面”或“负面”，不要解释。"


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    mode: Literal["base", "lora"] = "lora"


class PredictResponse(BaseModel):
    mode: str
    label: str | None
    raw_output: str
    latency_sec: float


class ModelSlot:
    def __init__(self, mode: str, base_model: str, adapter_path: str | None) -> None:
        self.mode = mode
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.adapter_path is not None and not Path(self.adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter not found: {self.adapter_path}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
        )
        if self.adapter_path is not None:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, text: str) -> PredictResponse:
        import torch

        self.load()
        assert self.tokenizer is not None
        assert self.model is not None
        prompt = build_prompt(self.tokenizer, INSTRUCTION, text)
        with torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            started = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return PredictResponse(
            mode=self.mode,
            label=extract_label(raw),
            raw_output=raw,
            latency_sec=latency,
        )


slots = {
    "base": ModelSlot("base", BASE_MODEL, None),
    "lora": ModelSlot("lora", BASE_MODEL, ADAPTER_PATH),
}

app = FastAPI(title="ChnSentiCorp LoRA Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    import torch

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "adapter_exists": Path(ADAPTER_PATH).exists(),
        "metrics": metrics,
    }


@app.post("/api/predict")
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return slots[request.mode].predict(request.text.strip())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve sentiment LoRA web UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
