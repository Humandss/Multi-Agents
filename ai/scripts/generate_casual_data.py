"""캐주얼 챗 시드를 베이스 EXAONE으로 자동 증강.

casual_seeds.yaml의 (input, output) 쌍을 보고, 같은 카테고리/캐릭터 어조로
N개의 변형을 생성. 출력은 ai/data/raw/{character}_casual.jsonl.

사용:
    uv run python scripts/generate_casual_data.py            # 5종 전체, 시드당 5개
    uv run python scripts/generate_casual_data.py --char hermann --per_seed 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
SEED_PATH = ROOT / "data" / "seed" / "casual_seeds.yaml"
RAW_DIR = ROOT / "data" / "raw"

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"

EXPAND_PROMPT = """게임 NPC 대화 데이터 변형 작업입니다.

NPC: {character} ({role_hint})
카테고리: {category}

원본 예시:
플레이어: {seed_input}
NPC: {seed_output}

위 NPC의 어조·말투·길이를 그대로 유지하면서, 같은 의미·기능의 다른 표현을 한 줄 만들어주세요.
플레이어 발화도 살짝 변형해도 됩니다(반말/존댓말, 다른 표현).
한 줄, JSON 형식: {{"input": "...", "output": "..."}}
다른 설명 없이 JSON만 출력.
"""

ROLE_HINTS = {
    "elias": "마법사. '흠.'으로 시작, '~소'/'~오' 어미, 학자적·회의적",
    "hermann": "대장장이. 단답·반말, '...' 자주 사용, 무뚝뚝하지만 내면 따뜻",
    "mathilda": "술집 주인. '어머~', '아유~', 따뜻한 어미, 사교적·다정",
    "finn": "음유시인. '오~', 시적 과장, '~지요/이옵니다'",
    "bernhardt": "잡화점 상인. '흠.', '~지요', '~이올시다', 정중·계산적",
}

META_REJECT = ["응답입니다", "물론입니다", "아래는", "다음과 같습니다", "JSON 형식", "예시:"]


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
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
    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, prompt, max_new_tokens=80):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    return text


def parse_json_line(text):
    text = text.strip().strip("`")
    # JSON 블록 추출
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        return None
    snippet = text[start:end + 1]
    try:
        obj = json.loads(snippet)
        if "input" in obj and "output" in obj:
            return {
                "input": str(obj["input"]).strip(),
                "output": str(obj["output"]).strip(),
            }
    except Exception:
        return None
    return None


def expand_for_character(character, character_seeds, per_seed, tokenizer, model):
    role_hint = ROLE_HINTS.get(character, "")
    out_rows = []

    for category, items in character_seeds.items():
        for seed_idx, seed in enumerate(items):
            prompt = EXPAND_PROMPT.format(
                character=character,
                role_hint=role_hint,
                category=category,
                seed_input=seed["input"],
                seed_output=seed["output"],
            )
            generated = 0
            attempts = 0
            while generated < per_seed and attempts < per_seed * 2:
                attempts += 1
                text = generate_one(tokenizer, model, prompt)
                # 메타 텍스트 컷
                if any(m in text for m in META_REJECT):
                    continue
                parsed = parse_json_line(text)
                if parsed is None:
                    continue
                if not parsed["input"] or not parsed["output"]:
                    continue
                if len(parsed["output"]) > 180:
                    continue
                # 시드와 동일하면 스킵
                if parsed["input"] == seed["input"] and parsed["output"] == seed["output"]:
                    continue
                out_rows.append({
                    "character": character,
                    "category": f"{category}_aug",
                    "input": parsed["input"],
                    "output": parsed["output"],
                })
                generated += 1
            print(f"  [{character}/{category}/seed{seed_idx}] {generated}/{per_seed} 생성")

        # 시드 자체도 학습 데이터에 포함
        for seed in items:
            out_rows.append({
                "character": character,
                "category": category,
                "input": seed["input"],
                "output": seed["output"],
            })

    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", choices=CHARACTERS, default=None)
    parser.add_argument("--per_seed", type=int, default=5, help="시드당 생성 변형 개수")
    args = parser.parse_args()

    print("[setup] base EXAONE 로딩...")
    tokenizer, model = load_model()

    with SEED_PATH.open(encoding="utf-8") as f:
        all_seeds = yaml.safe_load(f)

    targets = [args.char] if args.char else CHARACTERS
    for character in targets:
        if character not in all_seeds:
            print(f"  {character}: 시드 없음, skip")
            continue
        print(f"\n=== {character} 증강 시작 ===")
        rows = expand_for_character(
            character, all_seeds[character], args.per_seed, tokenizer, model
        )
        out_path = RAW_DIR / f"{character}_casual.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{character}: {len(rows)}개 → {out_path}")


if __name__ == "__main__":
    main()
