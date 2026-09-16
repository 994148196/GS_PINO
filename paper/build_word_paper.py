#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the Chinese Word (docx) version of paper/manuscript.md.

Follows the CNNC conference formatting requirements (notice 0901):
- standard academic layout: title / author / affiliation / abstract / keywords /
  body / figures / tables / references
- Chinese + English abstract, GB/T 7714 references
- native Word (OMML) equations, centered with right-aligned numbers
- A4, 2.5 cm margins, Song/SimHei + Times New Roman fonts

Output: paper/manuscript_cn.docx  (temp files in paper/_build_tmp, removed)
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import docx
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm

ROOT = Path(__file__).resolve().parent          # paper/
TMP = ROOT / "_build_tmp"
OUT = ROOT / "manuscript_cn.docx"
PANDOC = "pandoc"

TITLE_CN = "基于物理信息神经算子的 Grad-Shafranov 方程快速求解方法研究"
AUTHOR_CN = "陈佳林¹"
AFFIL_CN = "（1. 中国聚变能源有限公司，上海　201102）"
AUTHOR_EN = "Jialin Chen¹"
AFFIL_EN = "(1. China Fusion Energy Co., Ltd., Shanghai 201102, China)"
CLC_LINE = "**中图分类号**：TL612　　**文献标志码**：A"

# display-equation numbers in document order (None = continuation line, no number)
EQ_NUMS = [1, 2, None, 3, None, 4, 5, 6, 7, 8, None, 9]

REFS = """[1] Grad H, Rubin H. Hydromagnetic equilibria and force-free fields[C]//Proceedings of the 2nd UN Conference on the Peaceful Uses of Atomic Energy. Geneva: United Nations, 1958: 190-197.

[2] Shafranov V D. On magnetohydrodynamical equilibrium configurations[J]. Soviet Physics JETP, 1958, 6(3): 545-554.

[3] Lao L L, St John H, Stambaugh R D, et al. Reconstruction of current profile parameters and plasma shapes in tokamaks[J]. Nuclear Fusion, 1985, 25(11): 1611-1622.

[4] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.

[5] Li Z, Kovachki N, Azizzadenesheli K, et al. Fourier neural operator for parametric partial differential equations[C]//Proceedings of the International Conference on Learning Representations. 2021.

[6] Kovachki N, Li Z, Liu B, et al. Neural operator: Learning maps between function spaces[J]. Journal of Machine Learning Research, 2023, 24: 1-97.

[7] Joung S, Kim J, Kwak S, et al. Deep neural network Grad-Shafranov solver constrained with measured magnetic signals[J]. Nuclear Fusion, 2020, 60(1): 016034.

[8] Lu J, Hu Y, Xiang N, et al. Fast equilibrium reconstruction by deep learning on EAST tokamak[J]. AIP Advances, 2023, 13(7): 075007.

[9] Krastev P G. Millisecond-scale neural operator surrogates for double-null free-boundary Grad-Shafranov equilibria[EB/OL]. (2026)[2026-09-14]. https://arxiv.org/abs/2608.05555.

[10] Jeon Y M. Development of a free boundary tokamak equilibrium solver for advanced study of tokamak equilibria[J]. Journal of the Korean Physical Society, 2015, 67(5): 843-853.

[11] Han K S, Park B H, Aydemir A Y, et al. A free-boundary equilibrium solver with a hybrid iteration method in a semi-bounded computational domain[J]. Computer Physics Communications, 2021, 264: 107888.

[12] GS_solver: free-boundary Grad-Shafranov solver[CP/OL]. [2026-09-14]. https://github.com/994148196/GS_solver.

[13] Dudson B. Freegs: free boundary Grad-Shafranov solver[CP/OL]. [2026-09-14]. https://github.com/freegs-plasma/freegs.

[14] Bengio Y, Louradour J, Collobert R, et al. Curriculum learning[C]//Proceedings of the 26th Annual International Conference on Machine Learning. New York: ACM, 2009: 41-48.

[15] GS_PINO: physics-informed neural operator for free-boundary Grad-Shafranov equilibria[CP/OL]. [2026-09-14]. https://github.com/994148196/GS_PINO."""


def set_run_font(run, east, ascii_, size, bold=None):
    run.font.name = ascii_
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rpr.get_or_add_rFonts().set(qn("w:eastAsia"), east)


def set_style_font(style, east, ascii_, size, bold=None):
    style.font.name = ascii_
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    rpr.get_or_add_rFonts().set(qn("w:eastAsia"), east)


def build_markdown():
    src = (ROOT / "manuscript.md").read_text(encoding="utf-8")
    abstract_src = (ROOT / "abstract.md").read_text(encoding="utf-8")

    # ---- Chinese abstract & keywords from manuscript.md
    cn_abs = re.search(r"\*\*摘要\*\*：(.*?)\n\n\*\*关键词", src, re.S).group(1).strip()
    cn_kw = re.search(r"\*\*关键词\*\*：(.*?)\n", src).group(1).strip()

    # ---- English title / abstract / keywords from abstract.md
    en_title = re.search(r"## (A Physics-Informed.+)", abstract_src).group(1).strip()
    en_abs = re.search(r"\*\*Abstract\*\*: (.*?)\n\n", abstract_src, re.S).group(1).strip()
    en_kw = re.search(r"\*\*Key words\*\*: (.*?)\n", abstract_src).group(1).strip()
    # consistency fix: match the Chinese 8.6 ms figure
    en_abs = en_abs.replace("approximately 2 milliseconds", "approximately 8.6 milliseconds")

    # ---- body from manuscript.md
    body = src[src.index("## 1 引言"):]
    # drop trailing references (rebuilt below in GB/T 7714)
    body = body[:body.index("[1] Grad H")]
    body = re.sub(r"---\s*$", "", body.strip()) + "\n"

    # content normalization: 做法N -> 方案N (consistent with tables / English "scheme")
    body = body.replace("做法 1", "方案 1").replace("做法 2", "方案 2")
    cn_abs = cn_abs.replace("做法 1", "方案 1").replace("做法 2", "方案 2")

    # heading shift: ## -> #, ### -> ## (single pass, one level up)
    body = re.sub(r"(?m)^(#{1,6}) ", lambda m: "#" * max(1, len(m.group(1)) - 1) + " ", body)

    # internal anchor links -> plain text
    body = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", body)

    # strip \tag{n} (pandoc 2.12 texmath cannot convert it)
    body = re.sub(r"\s*\\tag\{\d+\}", "", body)

    # split aligned environments into stacked single-line display equations
    def split_aligned(m):
        rows = [r.replace("&", "").strip() for r in re.split(r"\\\\", m.group(1))]
        rows = [r for r in rows if r]
        return "\n\n".join(f"$${r}$$" for r in rows)
    body = re.sub(r"\$\$\\begin\{aligned\}(.*?)\\end\{aligned\}\$\$",
                  split_aligned, body, flags=re.S)

    # drop anchor lines, then figures: <p><img/></p> + <p><b>caption</b></p> -> pandoc implicit figure
    body = re.sub(r'<a id="[^"]*"></a>\n?', "", body)

    def fig_repl(m):
        return f"\n\n![{m.group(2)}](../{m.group(1)}){{width=15cm}}\n\n"
    body = re.sub(
        r'<p align="center">\s*<img src="([^"]+)"[^>]*>\s*</p>\s*<p align="center"><b>(.*?)</b></p>',
        fig_repl, body, flags=re.S)
    assert body.count("![") == 6, f"figures matched: {body.count('![')}"

    # table captions: move from before the pipe table to pandoc caption line after it
    def tab_repl(m):
        return m.group(2) + ": " + m.group(1) + "\n"
    body = re.sub(r'<p align="center"><b>(表\d[^<]*)</b></p>\n\n((?:\|[^\n]*\n)+)',
                  tab_repl, body)

    # superscript citations [n] / [n-m] / [n,m] in body (refs section excluded)
    body = re.sub(r"\[(\d+(?:[-,，]\d+)*)\]", r"^[\1]^", body)

    # 表2 note -> journal "注："
    body = body.replace("| 纯数据基线* |", "| 纯数据基线 |")
    body = re.sub(r"<small>\*\s*(.*?)</small>",
                  lambda m: "注：" + m.group(1).replace("* ", ""), body)
    body = body.replace("注：纯数据基线在 DN 与 SN 跨位形混合数据上训练（见 4.3 节），表中为其在双零位形测试集上的指标。",
                        "注：纯数据基线为 DN 与 SN 混合训练模型（见 4.3 节），表中为其在双零位形测试集上的指标。")

    # leftovers cleanup
    body = body.replace("&emsp;", "")
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"</?p[^>]*>", "", body)

    assert body.count("$$") == 2 * len(EQ_NUMS), f"equation count mismatch: {body.count('$$') // 2}"

    front = (f"{TITLE_CN}\n\n{AUTHOR_CN}\n\n{AFFIL_CN}\n\n"
             f"**摘　要**：{cn_abs}\n\n**关键词**：{cn_kw}\n\n{CLC_LINE}\n\n"
             f"{en_title}\n\n{AUTHOR_EN}\n\n{AFFIL_EN}\n\n"
             f"**Abstract**: {en_abs}\n\n**Key words**: {en_kw}\n\n")
    refs_md = "# 参考文献\n\n" + REFS + "\n"
    return front + body + "\n" + refs_md, cn_abs, body


def get_style(d, name):
    """Look up by reported style name (pandoc ref uses 'Heading 1', not the
    internal 'heading 1' that python-docx's __getitem__ maps to)."""
    for s in d.styles:
        if s.name == name:
            return s
    raise KeyError(name)


def make_reference_docx(path):
    subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"],
                   stdout=open(path, "wb"), check=True)
    d = Document(path)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, m, Cm(2.5))

    normal = get_style(d, "Normal")
    set_style_font(normal, "宋体", "Times New Roman", 10.5, bold=False)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name in ("Body Text", "First Paragraph"):
        get_style(d, name).paragraph_format.first_line_indent = Pt(21)
    compact = get_style(d, "Compact")
    set_style_font(compact, "宋体", "Times New Roman", 9, bold=False)
    compact.paragraph_format.line_spacing = 1.0
    compact.paragraph_format.first_line_indent = Pt(0)
    h1, h2 = get_style(d, "Heading 1"), get_style(d, "Heading 2")
    set_style_font(h1, "黑体", "Times New Roman", 14, bold=False)
    h1.paragraph_format.space_before, h1.paragraph_format.space_after = Pt(13), Pt(6.5)
    set_style_font(h2, "黑体", "Times New Roman", 12, bold=False)
    h2.paragraph_format.space_before, h2.paragraph_format.space_after = Pt(6.5), Pt(3)
    for name in ("Image Caption", "Table Caption"):
        st = get_style(d, name)
        set_style_font(st, "黑体", "Times New Roman", 9, bold=False)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.first_line_indent = Pt(0)
        st.paragraph_format.line_spacing = 1.2
        st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(6), Pt(6)
    fig = get_style(d, "Figure")
    fig.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig.paragraph_format.first_line_indent = Pt(0)
    d.save(path)


def postprocess(doc):
    # 0) ensure table captions sit above the table
    body_el = doc.element.body
    for tbl in body_el.findall(qn("w:tbl")):
        nxt = tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            style = nxt.find(qn("w:pPr") + "/" + qn("w:pStyle"))
            if style is not None and style.get(qn("w:val")) == "TableCaption":
                body_el.remove(nxt)
                tbl.addprevious(nxt)

    # 1) display equations -> centered math + right-aligned number via tab stops
    disp = [p for p in doc.paragraphs if p._element.find(qn("m:oMathPara")) is not None]
    assert len(disp) == len(EQ_NUMS), f"display eq paragraphs {len(disp)} != {len(EQ_NUMS)}"
    for p, num in zip(disp, EQ_NUMS):
        omp = p._element.find(qn("m:oMathPara"))
        om = omp.find(qn("m:oMath"))
        p._element.remove(omp)
        p._element.append(om)                      # oMathPara -> inline oMath
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.tab_stops.add_tab_stop(Cm(8.0), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT)
        lead = p.add_run()
        lead.add_tab()
        p._element.remove(lead._element)
        p._element.insert(1, lead._element)        # right after pPr
        if num is not None:
            r = p.add_run()
            r.add_tab()
            r.add_text(f"({num})")
            set_run_font(r, "宋体", "Times New Roman", 10.5)

    # 2) front matter
    consumed = {"title": False, "ac": False, "fc": False, "te": False,
                "ae": False, "fe": False}
    for p in doc.paragraphs:
        t = p.text.strip()
        pf = p.paragraph_format
        if not consumed["title"] and t.startswith("基于物理信息神经算子的"):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Pt(0)
            pf.space_after = Pt(10)
            for r in p.runs:
                set_run_font(r, "黑体", "Times New Roman", 16, bold=False)
            consumed["title"] = True
        elif not consumed["ac"] and t == "陈佳林¹":
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Pt(0)
            for r in p.runs:
                set_run_font(r, "宋体", "Times New Roman", 12)
            consumed["ac"] = True
        elif not consumed["fc"] and t.startswith("（1. 中国聚变能源"):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Pt(0)
            pf.space_after = Pt(12)
            for r in p.runs:
                set_run_font(r, "宋体", "Times New Roman", 9)
            consumed["fc"] = True
        elif not consumed["te"] and t.startswith("A Physics-Informed"):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Pt(0)
            pf.space_before, pf.space_after = Pt(12), Pt(8)
            for r in p.runs:
                set_run_font(r, "Times New Roman", "Times New Roman", 14, bold=True)
            consumed["te"] = True
        elif not consumed["ae"] and t == "Jialin Chen¹":
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Pt(0)
            for r in p.runs:
                set_run_font(r, "Times New Roman", "Times New Roman", 12)
            consumed["ae"] = True
        elif not consumed["fe"] and t.startswith("(1. China Fusion"):
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Pt(0)
            pf.space_after = Pt(12)
            for r in p.runs:
                set_run_font(r, "Times New Roman", "Times New Roman", 9)
            consumed["fe"] = True
    assert all(consumed.values()), f"front matter not fully matched: {consumed}"

    # 3) bold labels (摘要/关键词/...) in SimHei
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith(("摘　要", "关键词", "中图分类号")):
            for r in p.runs:
                if r.bold:
                    rpr = r._element.get_or_add_rPr()
                    rpr.get_or_add_rFonts().set(qn("w:eastAsia"), "黑体")

    # 4) references (after 参考文献 heading): 9 pt, hanging indent
    seen_refs = False
    n_refs = 0
    for p in doc.paragraphs:
        if p.text.strip() == "参考文献":
            seen_refs = True
            continue
        if seen_refs and p.text.strip():
            pf = p.paragraph_format
            pf.first_line_indent = Cm(-0.74)
            pf.left_indent = Cm(0.74)
            pf.line_spacing = 1.25
            pf.space_after = Pt(2)
            for r in p.runs:
                r.font.size = Pt(9)
            n_refs += 1
    assert n_refs == 15, f"refs {n_refs} != 15"

    # 5) tables: cells 9 pt already via Compact style; clear stray indents, single spacing
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Pt(0)
                    p.paragraph_format.line_spacing = 1.0
    # 6) the 表2 note
    for p in doc.paragraphs:
        if p.text.strip().startswith("注："):
            pf = p.paragraph_format
            pf.first_line_indent = Pt(0)
            pf.space_before = Pt(3)
            for r in p.runs:
                r.font.size = Pt(9)

    # 7) footer page number + core properties
    fp = doc.sections[0].footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r"PAGE \* MERGEFORMAT")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "18")
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    fp._element.append(fld)
    doc.core_properties.title = TITLE_CN
    doc.core_properties.author = "陈佳林"


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    md, cn_abs, body = build_markdown()
    md_path = TMP / "manuscript_cn.md"
    md_path.write_text(md, encoding="utf-8")

    ref = TMP / "ref.docx"
    make_reference_docx(ref)

    out_tmp = TMP / "out.docx"
    subprocess.run([PANDOC, "manuscript_cn.md", "-o", "out.docx",
                    "--reference-doc", "ref.docx"],
                   cwd=TMP, check=True)

    doc = Document(out_tmp)
    postprocess(doc)
    doc.save(OUT)

    # ---- validation summary
    cn_abs_n = len(re.findall(r"[一-鿿]", cn_abs))
    body_cn = re.sub(r"\$[^$]*\$|\[\^\d+[-,，\d]*\^]", "", body)
    body_n = len(re.findall(r"[一-鿿]", body_cn))
    n_tables = len(doc.tables)
    n_caps = sum(1 for p in doc.paragraphs if p.style.name == "Image Caption")
    n_tcaps = sum(1 for p in doc.paragraphs if p.style.name == "Table Caption")
    n_img = len(doc.inline_shapes)
    n_math = doc.element.xml.count("<m:oMath>")
    print(f"OK  -> {OUT}")
    print(f"size: {OUT.stat().st_size / 1e6:.1f} MB")
    print(f"CN chars: body={body_n} abstract={cn_abs_n} total={body_n + cn_abs_n} "
          f"(requirement 5000-8000)")
    print(f"tables={n_tables} table captions={n_tcaps} figures={n_img} "
          f"figure captions={n_caps} oMath elements={n_math}")
    if not (CN_BODY_MIN if False else 5000) <= body_n + cn_abs_n <= 8000:
        print("WARNING: CN char count outside 5000-8000")


if __name__ == "__main__":
    main()
