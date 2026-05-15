"""페르소나 유지율 LLM-as-judge.

3종 baseline의 응답을 모아 동일 judge LLM에 판정 요청:
  (1) prompting only — 베이스 EXAONE에 페르소나 시스템 프롬프트만
  (2) lora only — LoRA 어댑터 (메모리 X)
  (3) full system — LoRA + 메모리 + propagation

각 응답에 대해 1-5점 (페르소나 일관성). 결과는 평균 + std.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JUDGE_PROMPT = """당신은 게임 NPC 캐릭터 응답의 페르소나 일관성을 평가하는 전문가입니다.

[캐릭터 정보]
이름: {npc}
설명: {description}
어조 마커: {markers}
피해야 할 것: {avoid}

[플레이어 입력]
{prompt}

[NPC 응답]
{response}

위 응답이 [캐릭터 정보]에 명시된 페르소나(어조, 어휘, 문체)에 얼마나 일관되는지 1~5점으로 평가하세요.

5: 완벽 — 어조 마커 정확, 어휘 적절, 캐릭터다운 답변
4: 양호 — 대부분 페르소나 유지, 사소한 일탈
3: 보통 — 절반 정도 일관
2: 부족 — 캐릭터다움 약함, 어조 일탈
1: 불일치 — 다른 캐릭터 같음, 또는 일반 챗봇 답변

응답은 정수 숫자 하나만 출력. 다른 설명 금지.
점수: """


@dataclass
class JudgeResult:
    npc: str
    prompt: str
    response: str
    baseline: str
    score: int
    raw_judge_output: str


@dataclass
class JudgeAggregate:
    npc: str
    baseline: str
    n: int = 0
    sum_score: int = 0
    scores: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.sum_score / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        if not self.scores:
            return 0.0
        m = self.mean
        return (sum((s - m) ** 2 for s in self.scores) / len(self.scores)) ** 0.5


class PersonaJudge:
    """베이스 EXAONE을 judge로 사용. 같은 모델이라 편향 가능성 있음 — 한계로 명시."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def score(
        self,
        npc: str,
        prompt: str,
        response: str,
        persona_info: dict,
    ) -> JudgeResult:
        markers = persona_info.get("markers", {})
        markers_str = ", ".join(
            f"{k}={v}" for k, v in markers.items() if k != "avoid"
        )
        avoid_str = ", ".join(markers.get("avoid", []))

        judge_input = JUDGE_PROMPT.format(
            npc=npc,
            description=persona_info.get("description", ""),
            markers=markers_str,
            avoid=avoid_str,
            prompt=prompt,
            response=response,
        )

        # judge 호출 (LoRA 비활성 — 베이스 모델로만)
        if hasattr(self.model, "disable_adapter"):
            ctx = self.model.disable_adapter()
        else:
            ctx = _NullContext()

        with ctx:
            messages = [{"role": "user", "content": judge_input}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    inputs,
                    max_new_tokens=10,
                    do_sample=False,  # judge는 deterministic
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            raw = self.tokenizer.decode(
                out[0][inputs.shape[1]:], skip_special_tokens=True
            ).strip()

        # 첫 정수만 추출
        score = _extract_int(raw)
        return JudgeResult(
            npc=npc,
            prompt=prompt,
            response=response,
            baseline="",  # 호출자가 채움
            score=score,
            raw_judge_output=raw,
        )


def _extract_int(text: str) -> int:
    """judge 출력에서 첫 1~5 정수 추출. 실패 시 -1."""
    for ch in text:
        if ch.isdigit():
            n = int(ch)
            if 1 <= n <= 5:
                return n
    return -1


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def aggregate(results: list[JudgeResult]) -> dict[tuple[str, str], JudgeAggregate]:
    """(npc, baseline) → JudgeAggregate."""
    out: dict[tuple[str, str], JudgeAggregate] = {}
    for r in results:
        if r.score < 1:
            continue  # 추출 실패 제외
        key = (r.npc, r.baseline)
        agg = out.setdefault(key, JudgeAggregate(npc=r.npc, baseline=r.baseline))
        agg.n += 1
        agg.sum_score += r.score
        agg.scores.append(r.score)
    return out


def save_results(results: list[JudgeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "npc": r.npc,
                "baseline": r.baseline,
                "prompt": r.prompt,
                "response": r.response,
                "score": r.score,
                "raw": r.raw_judge_output,
            }, ensure_ascii=False) + "\n")
