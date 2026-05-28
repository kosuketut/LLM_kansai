#!/usr/bin/env python3
"""QLoRA SFT for Qwen3-Swallow-32B-CPT on a CUDA cluster.

Input format is the mlx-lm chat JSONL format used by this repository:
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
Only assistant tokens are used as labels, matching the local mlx-lm --mask-prompt
training setup.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL = "tokyotech-llm/Qwen3-Swallow-32B-CPT-v0.2"
DEFAULT_SYSTEM_PROMPT = (
    "あんたは大阪弁で話す気さくなアシスタントやで。"
    "どんな質問にも大阪弁で答えてな。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT for Osaka dialect tuning with Qwen3-Swallow-32B-CPT."
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="./data/osaka_data_v4")
    parser.add_argument("--output-dir", default="./outputs/qwen3_swallow_32b_cpt_qlora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=20)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA adapters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Use flash_attention_2 only when the container includes flash-attn.",
    )
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def data_files(data_dir: str | Path) -> dict[str, str]:
    root = Path(data_dir)
    files = {
        "train": root / "train.jsonl",
        "validation": root / "valid.jsonl",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {', '.join(missing)}")
    return {split: str(path) for split, path in files.items()}


def ensure_system_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *messages]


def find_last_assistant(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("Each training sample must contain at least one assistant message.")


def render_chat(tokenizer: AutoTokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        lines = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)


def preprocess_example(example: dict[str, Any], tokenizer: AutoTokenizer, max_seq_length: int) -> dict[str, list[int]]:
    messages = ensure_system_prompt(example["messages"])
    assistant_index = find_last_assistant(messages)
    prompt_messages = messages[:assistant_index]

    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
        input_ids.append(eos_id)

    if len(prompt_ids) >= max_seq_length:
        raise ValueError(
            "Prompt alone exceeds --max-seq-length. Increase max length or remove the sample."
        )

    input_ids = input_ids[:max_seq_length]
    labels = input_ids.copy()
    mask_until = min(len(prompt_ids), len(labels))
    labels[:mask_until] = [-100] * mask_until

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


@dataclass
class CausalLmDataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padding = [self.label_pad_token_id] * (max_length - len(label))
            padded_labels.append(label + padding)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    datasets = load_dataset("json", data_files=data_files(args.data_dir))
    tokenized = datasets.map(
        lambda example: preprocess_example(example, tokenizer, args.max_seq_length),
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing chat samples",
    )

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CausalLmDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
