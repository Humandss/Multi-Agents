"""정보 전파율 + 왜곡률 평가.

시나리오:
  1. 모든 NPC 시드 메모리만 있는 상태로 reset
  2. 한 NPC에 fact 주입 (예: mathilda에 곰 사건)
  3. N일(기본 7일) 시뮬
  4. 측정:
     - reach_ratio: 다른 4명 중 몇 명에게 정보가 도달했나
     - reach_by_day: 일별 누적 도달 비율
     - distortion_curve: 단계 거치며 의미 유사도 감소 곡선

사용:
    uv run python scripts/eval_propagation.py
    uv run python scripts/eval_propagation.py --days 7 --inject-to mathilda --fact "광장에 곰이 나타났다"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.distortion import BertDistortion, trace_distortion_from_events  # noqa: E402
from src.eval.propagation_rate import (  # noqa: E402
    compute_propagation_stats,
    filter_events_by_relevance,
)
from src.memory import MemoryEntry, MemorySource, MemoryStore  # noqa: E402
from src.propagation import PropagationSimulator, RelationGraph  # noqa: E402

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
ADAPTERS_DIR = ROOT / "output" / "adapters"
CHROMA_DIR = ROOT / "data" / "chroma"
RELATIONS_PATH = ROOT / "configs" / "relations.yaml"
RESULTS_DIR = ROOT / "output" / "eval"

import yaml as _yaml  # noqa


def load_base():
    cfg = _yaml.safe_load((ROOT / "configs" / "training.yaml").open(encoding="utf-8"))
    base_model = cfg["base_model"]
    base_revision = cfg.get("base_model_revision")

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
    return tokenizer, base


class _AdapterTransformer:
    """엔진과 같은 인터페이스를 흉내내는 가벼운 transformer."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}

    def transform(self, sender_npc: str, memory_text: str) -> str:
        key = (sender_npc, memory_text)
        if key in self.cache:
            return self.cache[key]
        self.model.set_adapter(sender_npc)
        prompt = (
            "다음 사실을 다른 마을 사람에게 한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
            f"\n사실: {memory_text}\n당신의 한 마디:"
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip().split("\n")[0].strip()
        self.cache[key] = text
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-to", default="mathilda", choices=CHARACTERS)
    parser.add_argument("--fact", default="광장에 큰 곰이 나타났다는 소식이 들어왔다.")
    parser.add_argument("--importance", type=int, default=8)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[eval/prop] inject {args.inject_to}: {args.fact}")
    print(f"           {args.days}일 시뮬, seed={args.seed}\n")

    # 1. 시드 + 주입
    stores = {}
    for npc in CHARACTERS:
        stores[npc] = MemoryStore(npc_name=npc, base_dir=CHROMA_DIR / npc)

    inject_id = f"inject_{uuid.uuid4().hex[:8]}"
    stores[args.inject_to].add(MemoryEntry(
        id=inject_id,
        text=args.fact,
        importance=args.importance,
        timestamp=datetime.now(timezone.utc),
        source=MemorySource.OBSERVATION,
    ))

    # 2. 베이스 + 어댑터 로드, transformer
    print("[eval/prop] 모델 로딩...")
    tokenizer, base = load_base()
    first = CHARACTERS[0]
    model = PeftModel.from_pretrained(base, str(ADAPTERS_DIR / first), adapter_name=first)
    for npc in CHARACTERS[1:]:
        model.load_adapter(str(ADAPTERS_DIR / npc), adapter_name=npc)
    model.eval()

    transformer = _AdapterTransformer(model, tokenizer)

    # 3. 시뮬
    graph = RelationGraph.load(RELATIONS_PATH)
    sim = PropagationSimulator(
        graph=graph, stores=stores, transformer=transformer, rng_seed=args.seed
    )

    all_events = []
    for d in range(1, args.days + 1):
        ev = sim.tick(d)
        all_events.extend(ev)
        print(f"  Day {d}: {len(ev)}건 전달 (누적 {len(all_events)})")

    # 4. 분석
    print("\n[측정] 의미 유사도 임베딩 모델 로딩...")
    embedder = BertDistortion()

    relevant_events = filter_events_by_relevance(all_events, args.fact, embedder, threshold=0.45)
    print(f"  전체 이벤트 {len(all_events)}, 그중 fact 관련 {len(relevant_events)}")

    # 전파율
    stats = compute_propagation_stats(args.inject_to, args.fact, relevant_events, CHARACTERS)
    print(f"\n[전파율]")
    print(f"  reach_ratio: {stats.reach_ratio*100:.0f}% ({len(stats.reached - {args.inject_to})}/{len(CHARACTERS) - 1}명 도달)")
    print(f"  first_reached_day:")
    for npc, day in sorted(stats.first_reached_day.items(), key=lambda x: x[1]):
        if npc != args.inject_to:
            print(f"    {npc:>10}: Day {day}")
    print(f"  reach_by_day:")
    for d in sorted(stats.reach_by_day.keys()):
        print(f"    Day {d}: {stats.reach_by_day[d]*100:.0f}%")

    # 왜곡 (각 단계의 변형 텍스트와 원본 비교)
    print(f"\n[왜곡률]")
    distortions = trace_distortion_from_events(relevant_events, args.fact, embedder)
    sims = [d.similarity for d in distortions]
    if sims:
        print(f"  평균 의미 유사도: {sum(sims)/len(sims):.3f}")
        print(f"  최저 (가장 변형됨): {min(sims):.3f}")
        print(f"  최고 (원본 가까움): {max(sims):.3f}")
        # sender별 평균
        per_sender = {}
        for d in distortions:
            per_sender.setdefault(d.sender, []).append(d.similarity)
        print(f"  sender별 평균 (페르소나 영향):")
        for sender, lst in sorted(per_sender.items()):
            print(f"    {sender:>10}: {sum(lst)/len(lst):.3f} (n={len(lst)})")

    # 결과 저장
    out_path = RESULTS_DIR / "propagation_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "events_total": len(all_events),
            "events_relevant": len(relevant_events),
            "reach_ratio": stats.reach_ratio,
            "first_reached_day": stats.first_reached_day,
            "reach_by_day": stats.reach_by_day,
            "distortion": {
                "avg_similarity": sum(sims) / len(sims) if sims else 0.0,
                "min_similarity": min(sims) if sims else 0.0,
                "max_similarity": max(sims) if sims else 0.0,
                "per_sender": {
                    s: sum(lst) / len(lst) for s, lst in per_sender.items()
                } if sims else {},
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
