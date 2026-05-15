"""정보 왜곡률 측정 (BERTScore).

원본 fact와 N단계 전파 후 NPC가 가진 메모리 텍스트를 비교.
페르소나에 따른 의도된 왜곡 vs 의도하지 않은 의미 손실 구분.

BERTScore는 사전학습된 BERT/RoBERTa 임베딩의 토큰 단위 cosine 유사도.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-m3"  # 한국어용 임베딩, 우리가 이미 받아둠


@dataclass
class DistortionResult:
    sender: str
    receiver: str
    original: str
    transformed: str
    similarity: float  # 0~1, 1에 가까울수록 의미 보존
    distortion: float  # 1 - similarity


class BertDistortion:
    """문장 임베딩 cosine 유사도로 정보 왜곡 측정.

    완전한 BERTScore (token-level F1)는 아니지만 sentence-level approximation으로
    충분히 의미 보존 정도를 측정 가능. 한국어 BGE-M3 사용.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.embedder = SentenceTransformer(model_name)

    def measure(self, original: str, transformed: str) -> float:
        embs = self.embedder.encode([original, transformed], normalize_embeddings=True)
        return float((embs[0] * embs[1]).sum())

    def measure_chain(self, original: str, propagation_chain: list[str]) -> list[float]:
        """원본 → 1단계 → 2단계 ... 각 단계와의 유사도 시퀀스."""
        return [self.measure(original, step) for step in propagation_chain]


def trace_distortion_from_events(
    events: list[dict],
    initial_fact: str,
    embedder: BertDistortion,
) -> list[DistortionResult]:
    """run_simulation 출력 events로부터 단계별 왜곡 추적.

    events: scripts/run_simulation.py --save-events 결과 (list of dict)
    """
    results = []
    for ev in events:
        sim = embedder.measure(initial_fact, ev["transformed"])
        results.append(DistortionResult(
            sender=ev["from"],
            receiver=ev["to"],
            original=ev["original"],
            transformed=ev["transformed"],
            similarity=sim,
            distortion=1.0 - sim,
        ))
    return results
