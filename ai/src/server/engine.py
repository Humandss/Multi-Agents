"""5종 NPC를 한 프로세스에서 추론하는 통합 엔진.

베이스 EXAONE 1개 + 5종 LoRA 어댑터를 PEFT의 load_adapter / set_adapter로
스위칭하면서 사용한다. 메모리 store/retriever도 NPC별로 보유.
또한 NPC 간 정보 전파(시간 기반)도 동일 프로세스에서 수행한다.
"""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..memory import MemoryEntry, MemoryRetriever, MemorySource, MemoryStore
from ..memory.chat import build_user_prompt
from ..propagation.graph import RelationGraph
from ..propagation.simulator import PropagationSimulator

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
BASE_REVISION = "496aef060b296b34c6b0035149f5af9e2b8c168c"

DEFAULT_CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]

# NPC별 generation 파라미터 차별화
# - hermann/elias: 짧고 무뚝뚝/회의적 → 낮은 temp + 짧은 max_tokens
# - mathilda/finn: 수다스럽고 시적 → 약간 높은 temp + 긴 max_tokens
# - bernhardt: 정중한 거래상 → 중간
GEN_PARAMS = {
    "hermann":   {"temperature": 0.35, "max_new_tokens": 100, "repetition_penalty": 1.20, "no_repeat_ngram_size": 4},
    "elias":     {"temperature": 0.35, "max_new_tokens": 130, "repetition_penalty": 1.20, "no_repeat_ngram_size": 4},
    "mathilda":  {"temperature": 0.50, "max_new_tokens": 160, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
    "finn":      {"temperature": 0.45, "max_new_tokens": 160, "repetition_penalty": 1.18, "no_repeat_ngram_size": 3},
    "bernhardt": {"temperature": 0.40, "max_new_tokens": 130, "repetition_penalty": 1.18, "no_repeat_ngram_size": 4},
}

PROMPT_FACT = (
    "다음 사실을 다른 마을 사람에게 한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "사람 이름과 장소 이름은 절대 바꾸지 말고, 어조만 너답게 바꾸세요. "
    "다른 설명이나 라벨은 붙이지 마세요.\n\n"
    "사실: {memory}\n\n"
    "당신의 한 마디:"
)

PROMPT_DIALOGUE = (
    "플레이어가 너에게 다음과 같이 말했다. 이 말에 담긴 사실 정보를 다른 마을 사람에게 "
    "한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "사람 이름과 장소 이름은 절대 바꾸지 말고, 어조만 너답게 바꾸세요. "
    "사실 정보가 없거나 단순 인사면 빈 답변을 출력하세요.\n\n"
    "플레이어 발언: {memory}\n\n"
    "당신의 한 마디:"
)


class NpcServer:
    def __init__(
        self,
        adapters_dir: Path,
        chroma_dir: Path,
        relations_path: Path | None = None,
        characters: list[str] | None = None,
        retrieval_k: int = 3,
    ):
        self.characters = characters or DEFAULT_CHARACTERS
        self.retrieval_k = retrieval_k

        adapter_paths = {npc: adapters_dir / npc for npc in self.characters}
        for npc, p in adapter_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"어댑터 없음: {p}")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("[engine] 토크나이저 + 베이스 모델 로딩...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        first = self.characters[0]
        print(f"[engine] LoRA 로딩 ({len(self.characters)}종)...")
        self.model = PeftModel.from_pretrained(
            base, str(adapter_paths[first]), adapter_name=first
        )
        for npc in self.characters[1:]:
            self.model.load_adapter(str(adapter_paths[npc]), adapter_name=npc)
        self.model.eval()

        print("[engine] 메모리 store/retriever 초기화...")
        self.stores: dict[str, MemoryStore] = {}
        self.retrievers: dict[str, MemoryRetriever] = {}
        for npc in self.characters:
            store = MemoryStore(npc_name=npc, base_dir=chroma_dir / npc)
            self.stores[npc] = store
            self.retrievers[npc] = MemoryRetriever(store)
            print(f"  {npc}: 메모리 {store.count()}개")

        # 정보 전파 그래프
        if relations_path is None:
            relations_path = Path(__file__).resolve().parents[2] / "configs" / "relations.yaml"
        if relations_path.exists():
            self.graph = RelationGraph.load(relations_path)
            print(f"[engine] 관계 그래프 로드 ({len(self.graph.edges())} edges)")
        else:
            self.graph = None
            print("[engine] 관계 그래프 없음, propagation 비활성")

        self.day = 0
        self._transform_cache: dict = {}

        # 페르소나 정의 로드 (system prompt에 사용)
        personas_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "test_prompts.yaml"
        try:
            with personas_path.open(encoding="utf-8") as f:
                self.personas = yaml.safe_load(f).get("personas", {})
            print(f"[engine] 페르소나 정의 로드 ({len(self.personas)}종)")
        except Exception as e:
            self.personas = {}
            print(f"[engine] 페르소나 정의 로드 실패: {e}")

    def _build_system_prompt(self, npc: str) -> str:
        """NPC별 system prompt — 페르소나 마커 + 영어 환각 방지 + 회상 활용 안내."""
        p = self.personas.get(npc, {})
        desc = p.get("description", "")
        m = p.get("markers", {})
        tone = ", ".join(m.get("tone", []))
        avoid = ", ".join(m.get("avoid", []))

        return (
            f"당신은 {npc}입니다. {desc}\n"
            f"어조: {tone}.\n"
            f"피해야 할 것: {avoid}.\n"
            "사용자 메시지 앞 괄호 안에 당신이 떠올린 정보가 있다면, 그 정보를 자연스럽게 답에 녹이세요. "
            "한국어로만 답하시오. 영어 단어, 외국어 표현 절대 금지. "
            "캐릭터답게 짧고 자연스럽게."
        )

    # ---------- 응답 생성 ----------
    def respond(
        self,
        npc: str,
        user_text: str,
        history: list[dict] | None = None,
        max_new_tokens: int = 200,
    ) -> dict:
        if npc not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc}")

        t0 = time.time()
        # 같은 NPC와 대화 중에는 자기가 들은 플레이어 발화(DIALOGUE)를 회상하지 않음.
        # (전파를 거쳐 다른 NPC가 PROPAGATION으로 다시 받게 되면 그건 회상 가능)
        retrieved = self.retrievers[npc].search(
            user_text, k=self.retrieval_k, exclude_sources={"dialogue"}
        )
        augmented = build_user_prompt(retrieved, user_text)

        # NPC별 system prompt 주입 (history 없을 때만, 있으면 첫 system 유지)
        messages = list(history or [])
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": self._build_system_prompt(npc)})
        messages.append({"role": "user", "content": augmented})

        # NPC별 generation 파라미터
        gp = GEN_PARAMS.get(npc, {
            "temperature": 0.5, "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
        })

        self.model.set_adapter(npc)
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=gp["max_new_tokens"],
                do_sample=True,
                temperature=gp["temperature"],
                top_p=0.9,
                top_k=50,
                repetition_penalty=gp["repetition_penalty"],
                no_repeat_ngram_size=gp["no_repeat_ngram_size"],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        latency_ms = int((time.time() - t0) * 1000)

        # 플레이어 발화를 NPC의 DIALOGUE 메모리로 저장 (다음 tick에서 전파 후보)
        self._save_player_turn(npc, user_text)

        return {
            "npc": npc,
            "text": text,
            "memories_used": [
                {
                    "text": m["text"],
                    "importance": m["importance"],
                    "source": m["metadata"].get("source", "unknown"),
                }
                for m in retrieved
            ],
            "latency_ms": latency_ms,
        }

    def _save_player_turn(self, npc: str, user_text: str) -> None:
        text = user_text.strip()
        if len(text) < 8:
            return  # 인사·감탄사는 저장 X (전파 가치 없음)

        # 질문문 vs 평서문 분기 — 질문은 fact가 아니므로 전파 후보에서 제외 (importance < threshold)
        is_question = "?" in text or any(
            text.endswith(suf) for suf in ["어요", "지요", "나요", "까", "야"]
        )
        # 사실 보고 키워드: 있으면 전파 가치 ↑
        fact_kw = ["나타났", "사라졌", "잡았", "봤", "들었", "있었", "갔다", "왔다", "했다", "됐다", "당했"]
        has_fact = any(kw in text for kw in fact_kw)

        if is_question:
            importance = 4   # 전파 X, 자유 대화 컨텍스트로만 사용
        elif has_fact and len(text) >= 12:
            importance = 9   # 평서문 + 사실 키워드 → 시드보다 강하게 전파
        elif len(text) >= 30:
            importance = 7
        elif len(text) >= 15:
            importance = 6
        else:
            importance = 5

        entry = MemoryEntry(
            id=f"dlg_{uuid.uuid4().hex[:8]}",
            text=f"플레이어가 말했다: {text}",
            importance=importance,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.DIALOGUE,
            metadata={"player": True, "is_question": is_question, "has_fact": has_fact},
        )
        self.stores[npc].add(entry)

    # ---------- PropagationSimulator transformer 인터페이스 ----------
    def transform(
        self,
        sender_npc: str,
        memory_text: str,
        source: str = "observation",
        max_new_tokens: int = 80,
    ) -> str:
        """sender NPC의 어조로 메모리를 다시 표현 (정보 전파 시 사용).

        source가 'dialogue'면 플레이어 발언에서 사실 정보 추출용 prompt,
        그 외에는 사실 그대로 전달용 prompt 사용.
        """
        cache_key = (sender_npc, memory_text, source)
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]

        self.model.set_adapter(sender_npc)
        # 메모리 prefix 제거: 더 깨끗한 입력으로
        clean = memory_text
        if clean.startswith("플레이어가 말했다: "):
            clean = clean[len("플레이어가 말했다: "):]
        elif "한테 들었다: " in clean:
            clean = clean.split("한테 들었다: ", 1)[1]

        template = PROMPT_DIALOGUE if source == "dialogue" else PROMPT_FACT
        prompt = template.format(memory=clean)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        text = text.split("\n")[0].strip().strip('"').strip("'")
        if not text:
            text = clean  # fallback: 원문 그대로
        self._transform_cache[cache_key] = text
        return text

    # ---------- 시간 진행 (정보 전파 tick) ----------
    def tick(self, day: int | None = None) -> dict:
        """하루치 정보 전파 시뮬레이션 실행 + 이벤트 반환."""
        if self.graph is None:
            return {"day": self.day, "events": [], "error": "관계 그래프 없음"}
        if day is None:
            self.day += 1
            day = self.day
        else:
            self.day = day

        sim = PropagationSimulator(
            graph=self.graph,
            stores=self.stores,
            transformer=self,
        )
        events = sim.tick(day)
        return {"day": day, "events": events}

    def memory_counts(self) -> dict[str, int]:
        return {npc: self.stores[npc].count() for npc in self.characters}
