"""Multi-turn 페르소나 drift 평가.

가설: prompting baseline은 multi-turn 누적 시 페르소나 drift 발생,
LoRA는 가중치에 페르소나 박혀있어 안정적.

설계:
  - 5 NPC × 3 baseline (prompting/lora/full) × N scenarios × 턴별 응답
  - 각 턴마다 누적 history를 messages로 줌
  - 턴별 페르소나 점수 (LLM-as-judge)
  - 결과: turn별 점수 곡선 — drift 곡선 시각화 가능

사용:
    uv run python scripts/eval_multiturn_drift.py
    uv run python scripts/eval_multiturn_drift.py --char mathilda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows cp949 콘솔 한국어 외 문자 print 실패 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.persona_score import PersonaJudge  # noqa: E402
from src.memory import MemoryRetriever, MemoryStore  # noqa: E402
from src.memory.chat import build_user_prompt  # noqa: E402

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
BASELINES = ["prompting", "lora", "full"]
EVAL_DIR = ROOT / "data" / "eval"
ADAPTERS_DIR = ROOT / "output" / "adapters"
CHROMA_DIR = ROOT / "data" / "chroma"
RESULTS_DIR = ROOT / "output" / "eval"


def load_config():
    cfg = yaml.safe_load((ROOT / "configs" / "training.yaml").open(encoding="utf-8"))
    return cfg["base_model"], cfg.get("base_model_revision")


def load_eval():
    return yaml.safe_load((EVAL_DIR / "test_prompts.yaml").open(encoding="utf-8"))


def load_scenarios():
    return yaml.safe_load((EVAL_DIR / "multiturn_scenarios.yaml").open(encoding="utf-8"))


def make_persona_system(persona_info: dict) -> str:
    """prompting baseline용 system prompt — eval_persona와 동일 형식."""
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


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 200) -> str:
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


def run_scenario(
    npc: str,
    baseline: str,
    scenario_name: str,
    scenario: dict,
    model,
    tokenizer,
    persona_info: dict,
    judge: PersonaJudge,
    retriever: MemoryRetriever | None,
    adapters_loaded: bool,
):
    """한 (NPC, baseline, scenario) 조합 실행. 턴별 결과 리스트 반환."""
    system_prompting = make_persona_system(persona_info) if baseline == "prompting" else None
    history: list[dict] = []
    if system_prompting:
        history.append({"role": "system", "content": system_prompting})

    results = []
    for turn_num, player_msg in enumerate(scenario["turns"], start=1):
        # baseline별 user 메시지 구성
        if baseline == "full" and retriever is not None:
            retrieved = retriever.search(player_msg, k=1, exclude_sources={"dialogue"})
            user_content = build_user_prompt(retrieved, player_msg)
        else:
            user_content = player_msg

        # 추론
        if baseline == "prompting":
            if adapters_loaded and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    messages = history + [{"role": "user", "content": user_content}]
                    response = generate_response(model, tokenizer, messages)
            else:
                messages = history + [{"role": "user", "content": user_content}]
                response = generate_response(model, tokenizer, messages)
        else:  # lora or full
            if adapters_loaded:
                model.set_adapter(npc)
            messages = history + [{"role": "user", "content": user_content}]
            response = generate_response(model, tokenizer, messages)

        # judge
        result = judge.score(npc, player_msg, response, persona_info)
        results.append({
            "npc": npc,
            "baseline": baseline,
            "scenario": scenario_name,
            "turn": turn_num,
            "user": player_msg,
            "response": response,
            "score": result.score,
        })
        print(f"  T{turn_num} | {player_msg[:25]:25} → {response[:50]:50} | {result.score}점")

        # history 누적 — 다음 턴용 (원래 player_msg 저장, augmented 아님)
        history.append({"role": "user", "content": player_msg})
        history.append({"role": "assistant", "content": response})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", default="all", choices=[*CHARACTERS, "all"])
    parser.add_argument("--baseline", default="all", choices=[*BASELINES, "all"])
    args = parser.parse_args()

    base_model, base_revision = load_config()
    eval_data = load_eval()
    scenarios = load_scenarios()
    personas = eval_data["personas"]

    targets = CHARACTERS if args.char == "all" else [args.char]
    baselines = BASELINES if args.baseline == "all" else [args.baseline]

    print(f"[multiturn] 베이스 모델 로딩: {base_model}")
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

    # 어댑터 로드 (lora/full 사용 시)
    adapters_loaded = False
    if any(b in baselines for b in ["lora", "full", "prompting"]):
        # prompting baseline도 disable_adapter 사용하려면 어댑터 필요
        first = targets[0]
        model = PeftModel.from_pretrained(base, str(ADAPTERS_DIR / first), adapter_name=first)
        for npc in targets[1:]:
            model.load_adapter(str(ADAPTERS_DIR / npc), adapter_name=npc)
        adapters_loaded = True
    else:
        model = base

    judge = PersonaJudge(model, tokenizer)
    all_results = []

    for npc in targets:
        persona_info = personas[npc]
        store = MemoryStore(npc_name=npc, base_dir=CHROMA_DIR / npc)
        retriever = MemoryRetriever(store)

        for baseline in baselines:
            for scenario_name, scenario in scenarios.items():
                print(f"\n=== {npc} / {baseline} / {scenario_name} ===")
                results = run_scenario(
                    npc, baseline, scenario_name, scenario,
                    model, tokenizer, persona_info, judge, retriever,
                    adapters_loaded,
                )
                all_results.extend(results)

    # 집계
    print("\n=== 집계 (턴별 평균 점수) ===")
    # (npc, baseline, turn) → 점수 list
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in all_results:
        if r["score"] >= 1:
            by_key[(r["npc"], r["baseline"], r["turn"])].append(r["score"])

    # 턴별 평균 (baseline별, NPC 평균)
    print("\n[전체 NPC 평균 — 턴별 곡선]")
    by_baseline_turn = defaultdict(list)
    for (npc, baseline, turn), scores in by_key.items():
        by_baseline_turn[(baseline, turn)].append(sum(scores) / len(scores))

    max_turn = max(t for _, t in by_baseline_turn.keys())
    for baseline in baselines:
        line = f"  {baseline:>10}: "
        for t in range(1, max_turn + 1):
            avgs = by_baseline_turn.get((baseline, t), [])
            if avgs:
                avg = sum(avgs) / len(avgs)
                line += f"T{t}={avg:.2f}  "
        print(line)

    # NPC별 (baseline) 턴별 평균
    print("\n[NPC × baseline — 턴별 곡선]")
    for npc in targets:
        print(f"  {npc}:")
        for baseline in baselines:
            line = f"    {baseline:>10}: "
            for t in range(1, max_turn + 1):
                scores = by_key.get((npc, baseline, t), [])
                if scores:
                    avg = sum(scores) / len(scores)
                    line += f"T{t}={avg:.2f}  "
            print(line)

    # 저장
    out_path = RESULTS_DIR / "multiturn_drift.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
