"""원본 학습 데이터에 검색 컨텍스트를 결합한 RAG-aware 학습 데이터 생성.

원본 dialog 일부를 시드 메모리와 페어링하고, base EXAONE으로 응답을
재작성해서 "context-aware" 학습 예제를 만든다.

목적: LoRA가 user 프롬프트의 fact prefix를 무시하지 않고 활용하도록 학습.

출력: ai/data/raw/{char}_rag.jsonl (기존 {char}.jsonl과 같은 포맷)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
RAW_DIR = ROOT / "data" / "raw"
SEED_PATH = ROOT / "data" / "seed" / "memories.yaml"

INFO_CATEGORIES = {"info_share", "smalltalk", "advice", "info_about_others", "rumor_propagation"}

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"

REWRITE_PROMPT = """게임 NPC의 대화를 한 줄 다시 쓰는 작업.

규칙:
- [사실]을 한 번 짧게 인용
- [원래 응답]의 어투·말끝·길이를 그대로 유지
- 1~2문장으로 짧게
- "응답:", "물론입니다", "아래는" 같은 메타 표현 금지
- 응답 텍스트만 출력

예시 1)
[사실]: 100년 전 광산 사고로 22명이 매몰됐다.
[질문]: 광산 다시 열 수 있어요?
[원래 응답]: 흠. 학문적으로는 가능하나, 안전 결계가 필요하오. 위험 부담이 크오.
[새 응답]: 흠. 100년 전 22명이 매몰된 그곳 말이오? 학문적으로는 가능하나, 안전 결계가 필요하오. 위험 부담이 크오.

예시 2)
[사실]: 어제 광장에서 모험가가 떠났다. 헤르만의 검을 들고 용을 잡으러 간다 했다.
[질문]: 마을에 무슨 일 있어요?
[원래 응답]: 어머~ 어제 잘생긴 모험가가 우리 가게 앞을 지나갔어요!
[새 응답]: 어머~ 어제 광장에서 그 잘생긴 모험가가 떠났어요. 헤르만 아저씨 검 들고 용 잡으러 간대요!

이제 작성:
[사실]: {fact}
[질문]: {question}
[원래 응답]: {original}
[새 응답]: """


META_REJECT = ["응답입니다", "응답은 다음", "물론입니다", "아래는", "다음과 같습니다", "요청하신", "보강하는", "도우미"]



def load_raw(character):
    path = RAW_DIR / f"{character}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_seeds():
    with SEED_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_relevant_seed(question, seeds, embedder, seed_embeds):
    """질문과 의미 유사도 가장 높은 시드 1개 반환."""
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    sims = seed_embeds @ q_emb
    idx = int(sims.argmax())
    return seeds[idx]


def generate_response(model, tokenizer, fact, question, original, max_new_tokens=120):
    prompt = REWRITE_PROMPT.format(fact=fact, question=question, original=original)
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
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    # 종종 모델이 prefix를 다시 출력하는 경우 정리
    for prefix in ["[새 응답]:", "[새 응답]", "응답:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # 줄바꿈 첫 줄만
    text = text.split("\n")[0].strip().strip('"').strip("'")
    return text


def augment_one(character, n_samples, model, tokenizer, embedder, seeds_yaml, seed=42):
    rng = random.Random(seed)
    rows = load_raw(character)

    # 정보 관련 카테고리만
    info_rows = [r for r in rows if r.get("category") in INFO_CATEGORIES]
    if len(info_rows) < n_samples:
        n_samples = len(info_rows)
    sampled = rng.sample(info_rows, n_samples)

    char_seeds = seeds_yaml.get(character, [])
    if not char_seeds:
        print(f"  {character}: 시드 없음, skip")
        return []

    seed_texts = [s["text"] for s in char_seeds]
    seed_embeds = embedder.encode(seed_texts, normalize_embeddings=True)

    augmented = []
    rejected = 0
    for i, row in enumerate(sampled, start=1):
        question = row["input"]
        original = row["output"]
        fact = pick_relevant_seed(question, seed_texts, embedder, seed_embeds)

        # 최대 2회 재시도
        new_response = None
        for _ in range(2):
            cand = generate_response(model, tokenizer, fact, question, original)
            if not cand or len(cand) < 5 or len(cand) > 250:
                continue
            if any(meta in cand for meta in META_REJECT):
                continue
            new_response = cand
            break

        if new_response is None:
            rejected += 1
            continue

        new_input = f"({fact}) {question}"
        augmented.append({
            "character": character,
            "category": row.get("category", "info_share") + "_rag",
            "input": new_input,
            "output": new_response,
        })
        if i % 5 == 0 or i == n_samples:
            print(f"  {character}: {i}/{n_samples} (생성 {len(augmented)}, 폐기 {rejected})")
    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", choices=CHARACTERS, default=None)
    parser.add_argument("--n", type=int, default=30, help="캐릭터당 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("base EXAONE 로딩...")
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

    print("BGE-M3 로딩...")
    embedder = SentenceTransformer("BAAI/bge-m3")

    seeds_yaml = load_seeds()
    targets = [args.char] if args.char else CHARACTERS

    for char in targets:
        print(f"\n=== {char} 증강 시작 (n={args.n}) ===")
        augmented = augment_one(char, args.n, model, tokenizer, embedder, seeds_yaml, args.seed)
        out_path = RAW_DIR / f"{char}_rag.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for sample in augmented:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"{char}: {len(augmented)}개 → {out_path}")


if __name__ == "__main__":
    main()
