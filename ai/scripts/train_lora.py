"""캐릭터별 LoRA 어댑터 학습."""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "training.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_quant_config(cfg):
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg):
    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        task_type=lc["task_type"],
        bias="none",
    )


def train_character(character, cfg, override_epochs=None):
    base_model = cfg["base_model"]
    revision = cfg.get("base_model_revision")
    train_cfg = cfg["train"]

    data_path = ROOT / "data" / "processed" / f"{character}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"전처리된 데이터셋 없음: {data_path}\n"
            f"먼저 preprocess.py 실행."
        )

    output_dir = ROOT / "output" / "adapters" / character
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {character} 학습 시작 ===")
    print(f"base : {base_model}")
    print(f"data : {data_path}")
    print(f"out  : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        quantization_config=build_quant_config(cfg),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"]
    )

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=["messages", "category"])
    print(f"샘플 수: {len(dataset)}")

    epochs = override_epochs if override_epochs is not None else train_cfg["num_epochs"]

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        max_seq_length=train_cfg["max_seq_length"],
        packing=False,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=build_lora_config(cfg),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"{character} 완료 → {output_dir}")

    del trainer, model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", required=True, choices=[*CHARACTERS, "all"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    targets = CHARACTERS if args.char == "all" else [args.char]

    for char in targets:
        train_character(char, cfg, override_epochs=args.epochs)


if __name__ == "__main__":
    main()
