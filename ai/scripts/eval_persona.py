"""페르소나 유지율 평가 (LLM-as-judge).

3종 baseline × 5종 NPC × N개 프롬프트 → 점수 집계.

baseline:
  - prompting: 베이스 EXAONE + system 프롬프트만 (페르소나 묘사)
  - lora: LoRA 어댑터만 적용, 메모리 X
  - full: LoRA + 메모리 + RAG 형식

사용:
    uv run python scripts/eval_persona.py --char all
    uv run python scripts/eval_persona.py --char elias --baseline lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows cp949 콘솔에서 em dash 등 한국어 외 문자 print 실패 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.persona_score import PersonaJudge, aggregate, save_results  # noqa: E402
from src.memory import MemoryRetriever, MemoryStore  # noqa: E402
from src.memory.chat import build_user_prompt  # noqa: E402

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
BASELINES = ["prompting", "lora", "full"]
EVAL_DIR = ROOT / "data" / "eval"
ADAPTERS_DIR = ROOT / "output" / "adapters"
CHROMA_DIR = ROOT / "data" / "chroma"
RESULTS_DIR = ROOT / "output" / "eval"

import yaml as _yaml  # noqa


def load_config():
    cfg = _yaml.safe_load((ROOT / "configs" / "training.yaml").open(encoding="utf-8"))
    return cfg["base_model"], cfg.get("base_model_revision")


def load_eval_prompts():
    return yaml.safe_load((EVAL_DIR / "test_prompts.yaml").open(encoding="utf-8"))


def make_persona_system(persona_info: dict) -> str:
    """prompting baseline용 verbose system prompt — 페르소나 묘사 전체 포함."""
    desc = persona_info.get("description", "")
    markers = persona_info.get("markers", {})
    speech_start = markers.get("speech_start", [])
    tone = markers.get("tone", [])
    vocab = markers.get("vocabulary", [])
    return (
        f"당신은 게임 NPC 캐릭터입니다.\n"
        f"캐릭터 설명: {desc}\n"
        f"말투 시작 패턴: {', '.join(speech_start)}\n"
        f"전반적 어조: {', '.join(tone)}\n"
        f"자주 쓰는 어휘: {', '.join(vocab)}\n"
        f"이 캐릭터로서 자연스럽게 짧게 답하세요."
    )


def make_light_system(npc: str, persona_info: dict) -> str:
    """LoRA + light system prompt 결합용 — 학습 분포 안 깨고 reinforce.

    이전 verbose system prompt 추가 시 LoRA -0.17 떨어짐 (학습 분포 외).
    Light 버전은 NPC 이름 + 1개 마커 hint만 — minimal nudge.
    """
    markers = persona_info.get("markers", {})
    speech_start = markers.get("speech_start", [])
    hint = f'"{speech_start[0]}"' if speech_start else "캐릭터답게"
    return f"당신은 {npc}입니다. {hint} 같은 말투로 짧고 자연스럽게 답하세요."


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system: str | None = None,
    augmented: str | None = None,
    max_new_tokens: int = 200,
) -> str:
    user_text = augmented or prompt
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", default="all", choices=[*CHARACTERS, "all"])
    parser.add_argument("--baseline", default="all", choices=[*BASELINES, "all"])
    parser.add_argument("--n_per_category", type=int, default=2,
                        help="카테고리당 사용할 프롬프트 수")
    args = parser.parse_args()

    base_model, base_revision = load_config()
    eval_data = load_eval_prompts()
    shared_prompts = eval_data["shared"]
    personas = eval_data["personas"]

    targets = CHARACTERS if args.char == "all" else [args.char]
    baselines = BASELINES if args.baseline == "all" else [args.baseline]

    print(f"[eval] 베이스 모델 로딩: {base_model}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # baseline 'lora'/'full'에 필요한 어댑터 로드
    adapters_loaded = False
    if any(b in baselines for b in ["lora", "full"]):
        first = targets[0]
        model = PeftModel.from_pretrained(base, str(ADAPTERS_DIR / first), adapter_name=first)
        for npc in targets[1:]:
            model.load_adapter(str(ADAPTERS_DIR / npc), adapter_name=npc)
        adapters_loaded = True
    else:
        model = base

    judge = PersonaJudge(model, tokenizer)
    all_results = []

    # 카테고리별로 N개씩 샘플링
    test_prompts = []
    for cat, lst in shared_prompts.items():
        test_prompts.extend([(cat, p) for p in lst[: args.n_per_category]])

    for npc in targets:
        persona_info = personas[npc]
        system_for_prompting = make_persona_system(persona_info)

        # 메모리 store (full에서만 사용)
        store = MemoryStore(npc_name=npc, base_dir=CHROMA_DIR / npc)
        retriever = MemoryRetriever(store)

        for baseline in baselines:
            print(f"\n=== {npc} / {baseline} ===")
            for cat, prompt in test_prompts:
                # baseline별 응답 생성
                if baseline == "prompting":
                    if adapters_loaded and hasattr(model, "disable_adapter"):
                        with model.disable_adapter():
                            response = generate_response(
                                model, tokenizer, prompt, system=system_for_prompting
                            )
                    else:
                        response = generate_response(
                            model, tokenizer, prompt, system=system_for_prompting
                        )
                elif baseline == "lora":
                    if not adapters_loaded:
                        print(f"  baseline=lora 인데 어댑터 미로드, skip")
                        continue
                    model.set_adapter(npc)
                    # Light system prompt 추가 — 학습 분포 안 깨면서 마커 reinforce.
                    # verbose는 -0.17 떨어졌지만 light(1줄)는 효과 다를 가능성.
                    light_sys = make_light_system(npc, persona_info)
                    response = generate_response(
                        model, tokenizer, prompt, system=light_sys
                    )
                elif baseline == "full":
                    if not adapters_loaded:
                        continue
                    model.set_adapter(npc)
                    # k=1: 회상 컨텍스트 줄여 페르소나 안정화 (production engine.py default와 일치)
                    retrieved = retriever.search(prompt, k=1, exclude_sources={"dialogue"})
                    augmented = build_user_prompt(retrieved, prompt)
                    light_sys = make_light_system(npc, persona_info)
                    response = generate_response(
                        model, tokenizer, prompt,
                        system=light_sys, augmented=augmented,
                    )

                # judge
                result = judge.score(npc, prompt, response, persona_info)
                result.baseline = baseline
                all_results.append(result)
                print(f"  [{cat}] {prompt[:30]} → {response[:40]}... (점수 {result.score})")

    # 집계
    print("\n=== 집계 ===")
    aggs = aggregate(all_results)
    for (npc, baseline), agg in sorted(aggs.items()):
        print(f"  {npc:>10} / {baseline:>10}: 평균 {agg.mean:.2f} ± {agg.std:.2f} (n={agg.n})")

    out_path = RESULTS_DIR / "persona_scores.jsonl"
    save_results(all_results, out_path)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
