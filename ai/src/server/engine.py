"""5종 NPC를 한 프로세스에서 추론하는 통합 엔진.

베이스 EXAONE 1개 + 5종 LoRA 어댑터를 PEFT의 load_adapter / set_adapter로
스위칭하면서 사용한다. 메모리 store/retriever도 NPC별로 보유.

VRAM 사용량 (RTX 4070 Ti 12GB 기준):
  - 베이스 (4bit): ~2GB
  - LoRA 어댑터 5개: ~400MB
  - KV cache + 임베딩 모델: ~1GB
  - 합계: ~3.5GB → 충분히 동작
"""

import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..memory import MemoryRetriever, MemoryStore
from ..memory.chat import build_user_prompt

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"

DEFAULT_CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]


class NpcServer:
    def __init__(
        self,
        adapters_dir: Path,
        chroma_dir: Path,
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

        # 첫 어댑터를 default로, 나머지는 load_adapter
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

    def respond(
        self,
        npc: str,
        user_text: str,
        history: list[dict] | None = None,
        max_new_tokens: int = 200,
    ) -> dict:
        """단일 응답 생성.

        history: 이전 대화 turn들. [{"role": "user"|"assistant", "content": "..."}, ...]
                 None이면 첫 턴으로 처리. 캐릭터 어조와 맥락 유지를 위해 사용.
        """
        if npc not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc}")

        t0 = time.time()
        retrieved = self.retrievers[npc].search(user_text, k=self.retrieval_k)
        augmented = build_user_prompt(retrieved, user_text)

        # history는 원본 텍스트 유지 (fact prefix 없음), 현재 턴만 augmented
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
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        latency_ms = int((time.time() - t0) * 1000)

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
