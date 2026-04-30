"""학습된 LoRA 어댑터로 페르소나 테스트."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"

DEFAULT_PROMPTS = [
    "안녕하세요!",
    "용 사냥꾼 얘기 들었어요?",
    "검 한 자루 주세요.",
    "마을에 무슨 일 있어요?",
    "고마워요.",
]


def generate(model, tokenizer, prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", required=True, choices=CHARACTERS)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-adapter", action="store_true")
    args = parser.parse_args()

    adapter_path = ROOT / "output" / "adapters" / args.char
    if not args.no_adapter and not adapter_path.exists():
        print(f"어댑터 없음: {adapter_path}")
        return 1

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("모델 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if not args.no_adapter:
        print(f"어댑터 적용: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    label = args.char + (" (베이스)" if args.no_adapter else "")

    if args.interactive:
        print(f"\n{label} 대화 모드 (종료: 'quit')")
        while True:
            try:
                prompt = input("\n플레이어 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt.lower() in {"quit", "exit"}:
                break
            print(f"{label}: {generate(model, tokenizer, prompt)}")
        return 0

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    for p in prompts:
        print(f"\n>>> {p}")
        print(f"{label}: {generate(model, tokenizer, p)}")


if __name__ == "__main__":
    main()
