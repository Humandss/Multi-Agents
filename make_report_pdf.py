"""report_draft.md → 보고서 형식 PDF (한글 폰트 임베드 + 평가 그림 자동 삽입).

실행: (ai 프로젝트 env 사용)
    cd ai && python -m uv run --with reportlab python ../make_report_pdf.py

- 한글: 맑은 고딕(malgun.ttf) 등록 → 깨짐 없음
- '> 작성 노트' 블록쿼트는 보고서에서 제외 (작성자 지침이므로)
- 5.2/5.3/5.4 절 끝에 output/eval 의 그림 3장 자동 삽입
"""

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent
MD = ROOT / "report_draft.md"
FIG_DIR = ROOT / "ai" / "output" / "eval"
OUT = ROOT / "report_draft.pdf"
FONTS = Path("C:/Windows/Fonts")

# --- 한글 폰트 등록 ---
_reg, _bold = FONTS / "malgun.ttf", FONTS / "malgunbd.ttf"
if not _reg.exists():
    print(f"[오류] 한글 폰트 없음: {_reg}")
    sys.exit(1)
pdfmetrics.registerFont(TTFont("KR", str(_reg)))
pdfmetrics.registerFont(TTFont("KR-Bold", str(_bold if _bold.exists() else _reg)))
registerFontFamily("KR", normal="KR", bold="KR-Bold", italic="KR", boldItalic="KR-Bold")

# --- 스타일 ---
TITLE = ParagraphStyle("title", fontName="KR-Bold", fontSize=19, leading=26,
                       alignment=TA_CENTER, textColor=colors.HexColor("#1a1a2e"))
SUBTITLE = ParagraphStyle("subtitle", fontName="KR", fontSize=12.5, leading=18,
                          alignment=TA_CENTER, textColor=colors.HexColor("#444"))
H1 = ParagraphStyle("h1", fontName="KR-Bold", fontSize=15, leading=20,
                    spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"))
H2 = ParagraphStyle("h2", fontName="KR-Bold", fontSize=12.5, leading=17,
                    spaceBefore=11, spaceAfter=5, textColor=colors.HexColor("#16213e"))
H3 = ParagraphStyle("h3", fontName="KR-Bold", fontSize=11, leading=15,
                    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#0f3460"))
BODY = ParagraphStyle("body", fontName="KR", fontSize=10.3, leading=16, spaceAfter=5)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, spaceAfter=2)
CODE = ParagraphStyle("code", fontName="KR", fontSize=8.5, leading=12.5,
                      backColor=colors.HexColor("#f3f3f5"), borderPadding=6,
                      leftIndent=4, textColor=colors.HexColor("#222"))
CAPTION = ParagraphStyle("caption", fontName="KR", fontSize=9, leading=12,
                         alignment=TA_CENTER, textColor=colors.HexColor("#666"), spaceAfter=10)
CELL = ParagraphStyle("cell", fontName="KR", fontSize=9.3, leading=13)
CELL_H = ParagraphStyle("cellh", fontName="KR-Bold", fontSize=9.3, leading=13)

CONTENT_W = 16 * cm
FIGS = {  # 절 번호 → (그림 파일, 캡션)
    "5.2": ("diffusion.png", "그림 3. 정보 확산 곡선 — 전파 ON/OFF 대조"),
    "5.3": ("diffusion_sweep.png", "그림 4. 접촉 빈도별 정보 확산 곡선"),
    "5.4": ("diffusion_network.png", "그림 5. 관계망 위 전파 경로 (출처·도달 틱)"),
}


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', text)
    return text


def scaled_image(path, max_w=14 * cm, max_h=10.5 * cm):
    iw, ih = ImageReader(str(path)).getSize()
    w, h = max_w, ih * max_w / iw
    if h > max_h:
        h, w = max_h, iw * max_h / ih
    return Image(str(path), width=w, height=h)


def make_table(rows):
    header, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), CELL_H) for c in header]]
    for r in body:
        data.append([Paragraph(inline(c), CELL) for c in r])
    ncol = len(header)
    t = Table(data, colWidths=[CONTENT_W / ncol] * ncol, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8ef")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build_story(lines):
    story = []
    pending_fig = None  # (절번호) — 다음 헤딩에서 그림 삽입
    i, n = 0, len(lines)

    def flush_fig():
        nonlocal pending_fig
        if pending_fig and pending_fig in FIGS:
            fname, cap = FIGS[pending_fig]
            fpath = FIG_DIR / fname
            if fpath.exists():
                story.append(Spacer(1, 4))
                story.append(scaled_image(fpath))
                story.append(Paragraph(cap, CAPTION))
            else:
                story.append(Paragraph(f"[그림 누락: {fname} — eval_diffusion.py 실행 필요]", CAPTION))
        pending_fig = None

    title_done = False
    while i < n:
        line = lines[i].rstrip("\n")
        s = line.strip()

        # 코드펜스
        if s.startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            story.append(Preformatted("\n".join(buf), CODE))
            story.append(Spacer(1, 4))
            continue

        # 블록쿼트(작성 노트) → 통째로 건너뜀
        if s.startswith(">"):
            while i < n and lines[i].lstrip().startswith(">"):
                i += 1
            continue

        # 표
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):  # 구분행 제외
                    rows.append(cells)
                i += 1
            if rows:
                flush_fig()
                story.append(make_table(rows))
                story.append(Spacer(1, 8))
            continue

        # 수평선
        if s == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#ccc")))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # 헤딩
        if s.startswith("#"):
            m = re.match(r"^(#+)\s+(.*)$", s)
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and not title_done:
                story.append(Spacer(1, 3 * cm))
                story.append(Paragraph(inline(text), TITLE))
                title_done = True
                i += 1
                continue
            if level == 2 and text.startswith("—"):  # 부제 (제목 페이지)
                story.append(Spacer(1, 6))
                story.append(Paragraph(inline(text.lstrip("— ").strip()), SUBTITLE))
                story.append(PageBreak())
                i += 1
                continue
            flush_fig()
            sty = {1: H1, 2: H2, 3: H3}.get(level, H3)
            story.append(Paragraph(inline(text), sty))
            mm = re.match(r"^(\d+\.\d+)", text)  # 5.2 등 → 그림 예약
            if mm and mm.group(1) in FIGS:
                pending_fig = mm.group(1)
            i += 1
            continue

        # 이미지 ![alt](path)
        mi = re.match(r"^!\[.*?\]\((.+?)\)", s)
        if mi:
            p = (ROOT / mi.group(1))
            if p.exists():
                story.append(scaled_image(p))
            i += 1
            continue

        # 리스트
        if s.startswith("- "):
            while i < n and lines[i].strip().startswith("- "):
                item = lines[i].strip()[2:]
                story.append(Paragraph("•&nbsp;&nbsp;" + inline(item), BULLET))
                i += 1
            story.append(Spacer(1, 4))
            continue

        # 빈 줄
        if s == "":
            i += 1
            continue

        # 일반 문단
        story.append(Paragraph(inline(s), BODY))
        i += 1

    flush_fig()
    return story


def main():
    if not MD.exists():
        print(f"[오류] {MD} 없음")
        return
    lines = MD.read_text(encoding="utf-8").splitlines()
    story = build_story(lines)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="NPC 창발적 사회 행동 보고서",
    )
    doc.build(story)
    print(f"[완료] {OUT}")


if __name__ == "__main__":
    main()
