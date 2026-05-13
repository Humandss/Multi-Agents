"""LoRA 어댑터 + 메모리 검색 기반 NPC 응답 생성.

⚠️  DEPRECATED (2026-05-03): 측정 결과 LoRA가 prompting baseline에 짐.
    Production 서버는 src/server/engine.py의 NpcServer(use_lora=False) 사용.
    이 NpcChat 클래스는 LoRA 활성 standalone 추론 (test_inference.py 등) 시만 사용.
    `build_user_prompt` 함수는 NpcServer에서도 사용 — 폐기 X.
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .retriever import MemoryRetriever
from .store import MemoryStore

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
BASE_REVISION = "496aef060b296b34c6b0035149f5af9e2b8c168c"


def build_user_prompt(retrieved, user_text):
    """검색된 메모리를 자연어 prefix로 user 메시지에 녹임.

    LoRA가 학습 데이터(짧은 user-assistant 쌍)에 강하게 fit돼있어서
    구조화된 프롬프트(`[내가 알고 있는 사실]` 같은) 안에 정보를 넣으면 무시한다.
    그래서 검색된 사실을 자연어 평서문으로 붙여서 LoRA가 문맥으로 받아들이게 한다.

    개선:
    - 메모리 source별 자연어 정리 (플레이어 발화 → "전에 듣기로", propagation → 그대로)
    - 최대 3개까지, "; "로 구분
    - "떠올려보니 —" prefix로 회상 분위기

    NOTE (2026-05-03): 학습 데이터 RAG 형식 `(평서문1. 평서문2.) 질문`과 통일
    시도했으나 mathilda 평가에서 -0.25 떨어져서 revert. EXAONE의 instruction-following이
    강해서 두 형식 모두 처리 가능, 출처 prefix("X한테 들었다") 자체가 페르소나 응답에
    유익한 신호로 작용하는 듯.
    """
    if not retrieved:
        return user_text

    parts = []
    for m in retrieved[:3]:
        text = m["text"].strip()
        if text.startswith("플레이어가 말했다:"):
            content = text[len("플레이어가 말했다:"):].strip()
            parts.append(f"전에 듣기로 {content}")
        elif "와 대화:" in text or "한테 들었다:" in text:
            # NPC-NPC 대화나 propagation 메모리는 너무 길게 끌고 들어오지 않도록 단축
            parts.append(text[:80])
        else:
            parts.append(text)

    facts = "; ".join(parts)
    # 명확히 구조화: 회상 따로 + 질문 따로. LLM이 회상에 끌려가지 않고 질문에 집중하도록.
    return f"[참고 기억: {facts}]\n질문: {user_text}"


class NpcChat:
    """단일 NPC의 LoRA + 메모리 추론 엔진."""

    def __init__(
        self,
        npc_name: str,
        adapter_dir: Path,
        chroma_dir: Path,
        retrieval_k: int = 1,  # 3 → 1: 회상 컨텍스트 줄여 페르소나 안정화
    ):
        self.npc_name = npc_name
        self.retrieval_k = retrieval_k

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

        self.store = MemoryStore(npc_name=npc_name, base_dir=chroma_dir / npc_name)
        self.retriever = MemoryRetriever(self.store)

    def respond(self, user_text: str, max_new_tokens: int = 200, return_memories: bool = False):
        retrieved = self.retriever.search(user_text, k=self.retrieval_k)
        augmented = build_user_prompt(retrieved, user_text)

        messages = [{"role": "user", "content": augmented}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
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
        response = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()

        if return_memories:
            return response, retrieved
        return response
