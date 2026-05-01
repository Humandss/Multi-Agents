"""LoRA 어댑터 + 메모리 검색 기반 NPC 응답 생성."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .retriever import MemoryRetriever
from .store import MemoryStore

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"


def build_user_prompt(retrieved, user_text):
    """검색된 메모리를 자연어 prefix로 user 메시지에 녹임.

    LoRA가 학습 데이터(짧은 user-assistant 쌍)에 강하게 fit돼있어서
    구조화된 프롬프트(`[내가 알고 있는 사실]` 같은) 안에 정보를 넣으면 무시한다.
    그래서 검색된 사실을 자연어 평서문으로 붙여서 LoRA가 문맥으로 받아들이게 한다.
    """
    if not retrieved:
        return user_text

    facts = " ".join(m["text"] for m in retrieved[:2])
    return f"({facts}) {user_text}"


class NpcChat:
    """단일 NPC의 LoRA + 메모리 추론 엔진."""

    def __init__(
        self,
        npc_name: str,
        adapter_dir: Path,
        chroma_dir: Path,
        retrieval_k: int = 3,
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
