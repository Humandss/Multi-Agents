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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..memory import MemoryEntry, MemoryRetriever, MemorySource, MemoryStore
from ..memory.chat import build_user_prompt
from ..propagation.graph import RelationGraph
from ..propagation.simulator import PropagationSimulator

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
BASE_REVISION = "496aef060b296b34c6b0035149f5af9e2b8c168c"

DEFAULT_CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]

TRANSFORM_PROMPT = (
    "다음 사실을 다른 마을 사람에게 한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "다른 설명이나 라벨은 붙이지 마세요.\n\n"
    "사실: {memory}\n\n"
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
        self._transform_cache: dict[tuple[str, str], str] = {}

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

        messages = list(history or [])
        messages.append({"role": "user", "content": augmented})

        self.model.set_adapter(npc)
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
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
        # 길이 따라 importance 차등: 길수록 정보성 높다 가정
        if len(text) >= 30:
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
            metadata={"player": True},
        )
        self.stores[npc].add(entry)

    # ---------- PropagationSimulator transformer 인터페이스 ----------
    def transform(self, sender_npc: str, memory_text: str, max_new_tokens: int = 80) -> str:
        """sender NPC의 어조로 메모리를 다시 표현 (정보 전파 시 사용)."""
        cache_key = (sender_npc, memory_text)
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]

        self.model.set_adapter(sender_npc)
        prompt = TRANSFORM_PROMPT.format(memory=memory_text)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        text = text.split("\n")[0].strip().strip('"').strip("'")
        if not text:
            text = memory_text  # fallback
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
