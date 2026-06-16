"""한국게임학회 양식(2단) 학회 논문 PDF 생성 — LoRA 제거판.

원본(반욱현, 2026 춘계 '페르소나 LoRA…' 제안 논문) 양식을 따르되,
LoRA를 제거하고 실제 구현 시스템(prompting 페르소나 + 기억 + 정보 전파)과
정보 확산 측정 결과(eval_diffusion.py 산출)를 넣어 '제안→결과' 논문으로 갱신.

실행:
    cd ai && python -m uv run --with reportlab python ../make_paper_pdf.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, FrameBreak, Frame, Image,
                                NextPageTemplate, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "ai" / "output" / "eval"
OUT = ROOT / "paper_game_society.pdf"
FONTS = Path("C:/Windows/Fonts")

pdfmetrics.registerFont(TTFont("KR", str(FONTS / "malgun.ttf")))
pdfmetrics.registerFont(TTFont("KR-Bold", str(FONTS / "malgunbd.ttf")))
registerFontFamily("KR", normal="KR", bold="KR-Bold", italic="KR", boldItalic="KR-Bold")

# ── 스타일 ──
TITLE = ParagraphStyle("t", fontName="KR-Bold", fontSize=16, leading=21, alignment=TA_CENTER)
ENTITLE = ParagraphStyle("et", fontName="KR", fontSize=10.5, leading=14, alignment=TA_CENTER,
                         textColor=colors.HexColor("#333"))
AUTHOR = ParagraphStyle("au", fontName="KR", fontSize=10.5, leading=14, alignment=TA_CENTER)
AFFIL = ParagraphStyle("af", fontName="KR", fontSize=9, leading=12, alignment=TA_CENTER,
                       textColor=colors.HexColor("#444"))
ABSHEAD = ParagraphStyle("ah", fontName="KR-Bold", fontSize=10, leading=14, alignment=TA_CENTER,
                         spaceBefore=8, spaceAfter=3)
ABS = ParagraphStyle("abs", fontName="KR", fontSize=8.7, leading=12.6, alignment=TA_JUSTIFY,
                     leftIndent=10, rightIndent=10)
KEY = ParagraphStyle("key", fontName="KR", fontSize=8.5, leading=12, leftIndent=10, rightIndent=10,
                     textColor=colors.HexColor("#333"), spaceBefore=2)
H2 = ParagraphStyle("h2", fontName="KR-Bold", fontSize=11, leading=15, spaceBefore=9, spaceAfter=3)
H3 = ParagraphStyle("h3", fontName="KR-Bold", fontSize=9.6, leading=13, spaceBefore=6, spaceAfter=2)
BODY = ParagraphStyle("b", fontName="KR", fontSize=9, leading=13.5, alignment=TA_JUSTIFY,
                      firstLineIndent=10, spaceAfter=2)
CODE = ParagraphStyle("c", fontName="KR", fontSize=8, leading=11.5,
                      backColor=colors.HexColor("#f3f3f5"), borderPadding=4, leftIndent=2)
CAP = ParagraphStyle("cap", fontName="KR", fontSize=8.3, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555"), spaceBefore=2, spaceAfter=6)
CELL = ParagraphStyle("cell", fontName="KR", fontSize=8, leading=10.5)
CELLH = ParagraphStyle("cellh", fontName="KR-Bold", fontSize=8, leading=10.5, alignment=TA_CENTER)
REF = ParagraphStyle("ref", fontName="KR", fontSize=8.3, leading=11.5, leftIndent=12,
                     firstLineIndent=-12, spaceAfter=1)


def P(t, s=BODY):
    return Paragraph(t, s)


def fig(name, cap, w):
    p = FIG / name
    if not p.exists():
        return P(f"[그림 누락: {name}]", CAP)
    iw, ih = ImageReader(str(p)).getSize()
    return [Image(str(p), width=w, height=ih * w / iw), P(cap, CAP)]


def char_table(w):
    rows = [["캐릭터", "역할", "특징(전파 보정)"],
            ["대장장이", "정보 시작점", "사실 그대로 (×1.0)"],
            ["술집주인", "정보 허브", "사교적·널리 전달 (×0.95)"],
            ["음유시인", "정보 증폭", "과장·왜곡 (×1.15)"],
            ["상인", "정보 거래", "가치 선택적 (×0.9)"],
            ["마법사", "정보 검증", "회의적·지연 수용 (×0.7)"]]
    data = [[Paragraph(c, CELLH) for c in rows[0]]] + \
           [[Paragraph(c, CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * 0.26, w * 0.30, w * 0.44])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8ef")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ── 레이아웃(2단) ──
PW, PH = A4
LM = RM = 1.6 * cm
TM = BM = 1.7 * cm
GUT = 0.7 * cm
UW = PW - LM - RM
CW = (UW - GUT) / 2
TITLE_H = 7.4 * cm


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("KR", 8)
    canvas.setFillColor(colors.HexColor("#555"))
    canvas.drawCentredString(PW / 2, 0.9 * cm,
                             f"2026년 한국게임학회 춘계 학술발표대회 논문집  ❙ {doc.page}")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4, title="한국어 게임 NPC 마을 시뮬레이션")
    title_frame = Frame(LM, PH - TM - TITLE_H, UW, TITLE_H, id="title",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    ch1 = (PH - TM - TITLE_H - 0.3 * cm) - BM
    l1 = Frame(LM, BM, CW, ch1, id="l1", leftPadding=0, rightPadding=0)
    r1 = Frame(LM + CW + GUT, BM, CW, ch1, id="r1", leftPadding=0, rightPadding=0)
    ch = PH - TM - BM
    lf = Frame(LM, BM, CW, ch, id="l", leftPadding=0, rightPadding=0)
    rf = Frame(LM + CW + GUT, BM, CW, ch, id="r", leftPadding=0, rightPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[title_frame, l1, r1], onPage=footer),
        PageTemplate(id="rest", frames=[lf, rf], onPage=footer),
    ])

    s = []
    # ── 제목/저자/초록 (전단 폭) ──
    s += [P("멀티 에이전트 기억과 사회적 정보 전파를 활용한<br/>한국어 게임 NPC 마을 시뮬레이션", TITLE)]
    s += [Spacer(1, 5), P("반욱현", AUTHOR),
          P("홍익대학교 게임학부  dnrgusqks@naver.com", AFFIL)]
    s += [Spacer(1, 4),
          P("A Korean Game NPC Village Simulation Using Multi-Agent Memory "
            "and Social Information Propagation", ENTITLE),
          P("Uk-Hyeon Ban,  School of Games, Hongik University", AFFIL)]
    s += [P("요 약", ABSHEAD)]
    s += [P("본 논문은 게임 내 비플레이어 캐릭터(NPC)의 몰입감을 향상시키기 위해, 한국어 로컬 "
            "대규모 언어 모델을 기반으로 개별 기억과 NPC 간 정보 전파를 결합한 마을 시뮬레이션을 "
            "구현한다. 기존 분기형 대화 트리는 자유 입력에 유연하지 못하며, 동일 마을의 NPC들이 "
            "서로 단절되어 플레이어 행동이 게임 세계로 확산되지 못한다. 본 시스템은 EXAONE 3.5를 "
            "베이스로 각 NPC가 벡터 데이터베이스 기반 장기 기억(가중 검색·망각·성찰)을 보유하고, "
            "관계 그래프를 따라 정보가 전파·변형되어 소문과 평판이 형성되는 사회적 동역학을 구현한다. "
            "정보 확산율을 Park 외의 방식으로 측정한 결과, 전파를 활성화하면 정보가 마을 전체로 "
            "확산(접촉 빈도 0.9에서 1틱 내 전원 도달)되는 반면 비활성 시 출처 NPC에 고정되어 전파 "
            "시스템의 기여를 정량적으로 확인하였다. 전 과정은 소비자용 GPU에서 오프라인으로 동작한다.", ABS)]
    s += [P("핵심어 : 비플레이어 캐릭터, 대규모 언어 모델, 에이전트 기억, 정보 전파, 사회 시뮬레이션", KEY)]
    s += [NextPageTemplate("rest"), FrameBreak()]

    # ── 1. 서론 ──
    s += [P("1. 서 론", H2)]
    s += [P("게임에서 비플레이어 캐릭터(이하 NPC)는 플레이어의 몰입감을 결정하는 핵심 요소다. "
            "그러나 기존 상용 게임의 NPC 대화는 사전 작성된 분기 트리에 의해 결정되어 플레이어의 "
            "자유로운 입력에 유연하게 반응하지 못한다. 또한 동일한 마을의 NPC들이 서로 단절되어 "
            "있어, 플레이어의 행동이 다른 NPC에게 전달되거나 영향을 주지 못해 몰입감이 단절된다.")]
    s += [P("최근 대규모 언어 모델(Large Language Model, LLM)의 발전으로 자연스러운 NPC 대화 "
            "생성이 가능해졌으며, Mantella와 같은 모드는 상용 게임에 LLM 기반 NPC 대화를 적용한 "
            "사례를 보여주었다[1]. 그러나 단순 프롬프팅은 대화가 길어질수록 캐릭터성이 무너지는 "
            "페르소나 드리프트가 보고되고, 외부 API 의존은 비용과 지연을 유발한다. 한편 Park 외[2]의 "
            "Generative Agents는 LLM 에이전트들이 가상 마을에서 자율적으로 상호작용하며 사회적 "
            "행동을 형성함을 보였으나, 실제 게임 환경에 통합된 사례는 제한적이다.")]
    s += [P("본 논문은 다음을 기여한다. 첫째, 한국어 로컬 LLM 기반으로 각 NPC가 가중 검색·망각·성찰을 "
            "갖춘 개별 장기 기억을 보유하는 구조를 구현한다. 둘째, 관계 그래프를 따라 정보가 전파·변형되어 "
            "소문과 평판이 형성되는 마을 단위 사회적 동역학을 구현한다. 셋째, 정보 확산율을 정량 측정하여 "
            "전파 시스템의 기여를 입증하고, Unity 게임 엔진과 통합 가능한 아키텍처를 제시한다.")]

    # ── 2. 관련 연구 ──
    s += [P("2. 관련 연구", H2)]
    s += [P("2.1 LLM 기반 NPC 대화와 페르소나", H3)]
    s += [P("LLM에 페르소나를 부여하는 대표적 방식은 시스템 프롬프트에 캐릭터 설정을 명시하는 "
            "프롬프팅 기반이다. 구현이 간단하나 다중 턴 대화에서 페르소나가 점차 희석되는 한계가 "
            "관찰된다. 본 연구는 강한 페르소나 프롬프트와 한국어 후처리(어미·조사·호칭 교정)를 결합하여 "
            "별도 파인튜닝 없이도 캐릭터 일관성을 확보한다.")]
    s += [P("2.2 에이전트 기억과 멀티 에이전트 시뮬레이션", H3)]
    s += [P("Park 외[2]는 25명의 생성형 에이전트가 관찰·반성(reflection)·계획으로 구성된 기억 구조를 "
            "통해 자율적으로 상호작용함을 보였다. 이러한 기억은 Lewis 외[3]의 검색 증강 생성(RAG)에 "
            "기반하며 벡터 데이터베이스를 활용한 의미 기반 회상이 핵심이다. 또한 Zhong 외[4]의 "
            "MemoryBank는 Ebbinghaus 망각 곡선을 LLM 기억에 적용하여 시간에 따른 망각을 모델링하였고, "
            "본 연구의 망각 메커니즘은 이 계보에 있다.")]
    s += [P("2.3 한국어 로컬 LLM", H3)]
    s += [P("LG AI Research의 EXAONE 3.5[5]는 한국어 처리에서 우수한 성능을 보고하였으며, 경량 "
            "모델을 제공하여 소비자용 GPU에서의 로컬 추론에 적합하다. 본 연구는 이를 베이스 모델로 "
            "활용하여 외부 API 없이 오프라인 동작을 달성한다.")]

    # ── 3. 제안 시스템 ──
    s += [P("3. 제안 시스템", H2)]
    s += [P("3.1 시스템 구조", H3)]
    s += [P("제안 시스템은 네 모듈로 구성된다. (1) 시스템 프롬프트 기반 페르소나로 동작하는 5종 NPC, "
            "(2) 각 NPC의 경험을 사건 단위로 저장·회상하는 벡터 메모리, (3) NPC 간 정보 전파를 담당하는 "
            "시뮬레이션 모듈, (4) Unity 게임 엔진과의 통신 인터페이스이다. 추론은 Python 서버(FastAPI, "
            "WebSocket)에서 수행되며 5종 NPC가 하나의 모델을 공유한다.")]
    s += [P("3.2 캐릭터 페르소나 설계", H3)]
    s += [P("5종 NPC는 각각 마을 내 정보 전파에서 서로 다른 역할을 수행하도록 설계된다(표 1). "
            "대장장이는 정보의 출발점으로 사실 위주로 전달하고, 술집주인은 다수와 연결된 허브다. "
            "음유시인은 정보를 증폭·과장하며, 상인은 가치로 환산해 선택적으로 보존하고, 마법사는 "
            "회의적 검증자로 정보를 의심하며 늦게 수용한다. 이 역할은 전파 시 중요도 보정 계수로 구현된다.")]
    s += [char_table(CW), P("[표 1] NPC 캐릭터 설계 및 전파 역할", CAP)]
    s += [P("3.3 기억 구조", H3)]
    s += [P("각 NPC는 ChromaDB와 BGE-M3 임베딩 기반의 개별 기억을 보유한다. 회상은 의미 유사도에 "
            "중요도·최신성을 가중하여 수행한다. 일화적 기억은 시간에 따라 잊히며, 보존 강도는 망각 "
            "곡선을 따른다:")]
    s += [Preformatted("retention = exp( -Δd / (3.0 × (1 + recall_count)) )", CODE)]
    s += [P("여기서 Δd는 마지막 접근 이후 경과 일수, recall_count는 회상 횟수로, 자주 떠올린 기억일수록 "
            "천천히 잊힌다. 기억이 누적되면 NPC는 상위 통찰을 생성하는 성찰(reflection)을 수행한다.")]
    s += [P("3.4 정보 전파", H3)]
    s += [P("시간 진행(틱)마다 관계 그래프의 각 엣지에 대해 만남이 발생하고, 보내는 NPC의 기억 중 "
            "중요도가 임계값 이상인 것이 받는 NPC에게 \"(상대)에게서 들었다\" 형태로 저장된다. 중요도는 "
            "전달자의 성격에 따라 보정되며(표 1), 정보의 최초 출처를 추적한다. 한 틱 안에서 정보를 받은 "
            "NPC가 다시 전달자가 될 수 있어 다단계 전파가 일어난다. 그 결과 소문의 확산·변형과 플레이어 "
            "평판이 별도 설계 없이 형성된다.")]

    # ── 4. 평가 ──
    s += [P("4. 평가: 정보 확산율", H2)]
    s += [P("Park 외[2]가 특정 정보를 한 에이전트에 심은 뒤 확산 정도를 측정한 방식을 따른다. 한 NPC에 "
            "사실을 주입하고 시간을 진행시키며, 각 틱마다 그 정보를 아는 NPC 수를 각 NPC의 기억에서 "
            "직접 집계한다. 한 틱은 하루에 해당하며 하루 안에 정보가 여러 사람을 거칠 수 있다.")]
    s += [P("4.1 전파의 기여", H3)]
    s += [P("전파의 활성/비활성을 비교한 결과, 접촉 빈도 0.9(데모 설정)에서 정보는 1틱 내 5명 전원에게 "
            "도달한 반면, 전파를 비활성화하면 출처 NPC 1명에 고정되었다. 이는 소문·평판이 전파 "
            "메커니즘에서 비롯됨을 정량적으로 보여준다.")]
    s += [P("4.2 접촉 빈도에 따른 확산 동역학", H3)]
    s += [P("동일 위상에서 접촉 빈도만 변화시켜 확산 동역학을 관찰하였다(그림 1). 빈도 0.9/0.5/0.3/0.2에서 "
            "전원 도달까지 각각 1/2/3/3틱이 소요되어, 접촉 빈도가 높을수록 한 틱에 완성되는 전파 사슬이 "
            "길어져 확산이 빨라짐을 확인하였다.")]
    s += fig("diffusion_sweep.png", "[그림 1] 접촉 빈도별 정보 확산 곡선", CW)
    s += [P("4.3 전파 경로와 관계망", H3)]
    s += [P("정보의 전달 경로를 추적한 결과(그림 2), 접촉 빈도 0.3·출처 hermann 기준 경로는 "
            "hermann→bernhardt→mathilda→{elias, finn}이었다. 모든 정보의 출처는 hermann으로 "
            "추적되며, 도달 순서가 관계 그래프의 위상을 반영하여 정보가 무작위가 아니라 사회관계 구조를 "
            "따라 흐름을 보여준다.")]
    s += fig("diffusion_network.png", "[그림 2] 관계망 위 전파 경로(출처·도달 틱)", CW)

    # ── 5. 결론 ──
    s += [P("5. 결론 및 향후 과제", H2)]
    s += [P("본 논문은 한국어 로컬 LLM 위에 개별 기억(검색·망각·성찰)과 NPC 간 정보 전파를 결합하여, "
            "소문과 평판이 형성되는 마을 NPC 시뮬레이션을 구현하였다. 정보 확산율 측정으로 전파 시스템의 "
            "기여를 정량적으로 확인하였고, 전 과정이 오프라인으로 동작함을 보였다. 향후 전파 시 내용 변형 "
            "품질 개선, 페르소나 유지율·정보 왜곡률의 정량 평가, 대규모 마을로의 확장을 과제로 한다.")]

    # ── 참고문헌 ──
    s += [P("참고문헌", H2)]
    refs = [
        "[1] Mantella, \"LLM-based dynamic NPC dialogue mod,\" 2023.",
        "[2] J. S. Park et al., \"Generative Agents: Interactive Simulacra of Human Behavior,\" UIST, 2023.",
        "[3] P. Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,\" NeurIPS, 2020.",
        "[4] W. Zhong et al., \"MemoryBank: Enhancing LLMs with Long-Term Memory,\" AAAI, 2024.",
        "[5] LG AI Research, \"EXAONE 3.5 Technical Report,\" 2024.",
    ]
    for r in refs:
        s.append(P(r, REF))

    doc.build(s)
    print(f"[완료] {OUT}")


if __name__ == "__main__":
    build()
