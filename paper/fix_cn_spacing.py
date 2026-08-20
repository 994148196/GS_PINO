"""CN-EN spacing normalizer for the paper markdown sources (pangu-style).

Rules (applied OUTSIDE protected regions only):
  1. space between CJK and ASCII letters/digits      (相对L2误差 -> 相对 L2 误差)
  2. space between CJK and %/×/²/³ symbols           (0.84%降至 -> 0.84% 降至)
Protected verbatim: $$...$$, $...$, <...> HTML tags,
[text](#anchor) markdown links, and caption numbers 图N　/ 表N　.

Usage: python fix_cn_spacing.py [file.md ...]
(no args -> paper/manuscript.md and paper/abstract.md)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = [ROOT / "paper" / "manuscript.md", ROOT / "paper" / "abstract.md"]

HAN = r"一-鿿"
SYMS = r"%×²³"  # % × ² ³

tokens: list[str] = []


def protect(m: re.Match) -> str:
    tokens.append(m.group(0))
    return f"\x00{len(tokens) - 1}\x00"


def run(path: Path) -> int:
    global tokens
    tokens = []
    t = path.read_text(encoding="utf-8")

    # 1. protect regions that must not be touched
    t = re.sub(r"\$\$[^$]*\$\$", protect, t)        # display math
    t = re.sub(r"\$[^$\n]+\$", protect, t)          # inline math
    t = re.sub(r"<[^>]+>", protect, t)              # HTML tags
    t = re.sub(r"\[[^\]]*\]\(#[^)]*\)", protect, t)  # markdown anchor links
    t = re.sub(rf"([图表])\d(?=　)", protect, t)     # caption numbers 图N　/ 表N

    # 2. spacing rules
    n = 0
    for pat in (
        rf"(?<=[{HAN}])(?=[A-Za-z0-9])",     # 汉字 → 字母/数字
        rf"(?<=[A-Za-z0-9])(?=[{HAN}])",     # 字母/数字 → 汉字
        rf"(?<=[{HAN}])(?=[{SYMS}])",        # 汉字 → 符号
        rf"(?<=[{SYMS}])(?=[{HAN}])",        # 符号 → 汉字
    ):
        t, k = re.subn(pat, " ", t)
        n += k

    # 3. restore protected tokens
    t = re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], t)
    path.write_text(t, encoding="utf-8")
    return n


INDENT = "&emsp;&emsp;"  # 2 EM SPACES ≈ 2 CJK char widths, never collapsed


def indent_paragraphs(path: Path) -> int:
    """Prepend "&emsp;&emsp;" to every paragraph's first line.

    EM SPACE (U+2003) is not in the collapsible white-space set, so the
    indent survives in any markdown->HTML pipeline even when embedded
    <style> rules are ignored; the entity form is visible in the source.
    Skipped verbatim: frontmatter, <style> blocks, HTML comment blocks,
    display-math blocks, headings, tables, HTML lines, the reference list
    (after <h2 id="refs">), and continuation lines of multi-line
    paragraphs. Idempotent: strips any previous leading U+3000 / nbsp /
    em-space characters or &emsp; entities first.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    lines = [re.sub(r"^(?:[　  ]|&emsp;)+", "", ln) for ln in lines]
    out: list[str] = []
    in_front = seen_front = False
    in_style = in_math = in_comment = in_refs = in_div = in_para = False
    n = 0
    for ln in lines:
        s = ln.strip()
        if not seen_front and s == "---":      # YAML frontmatter start
            seen_front = in_front = True
            out.append(ln); continue
        if in_front:
            if s == "---":
                in_front = False
            out.append(ln); continue
        if s == "<style>":
            in_style = True
            out.append(ln); continue
        if in_style:
            if s == "</style>":
                in_style = False
            out.append(ln); continue
        if "<div" in s:                           # centered div block (title/author)
            in_div = True
            out.append(ln); continue
        if in_div:
            if "</div>" in s:
                in_div = False
            out.append(ln); continue
        if s.startswith("<!--"):                 # HTML comment block
            in_comment = not s.endswith("-->")
            out.append(ln); continue
        if in_comment:
            if s.endswith("-->"):
                in_comment = False
            out.append(ln); continue
        if s.startswith("$$"):                   # display-math block
            in_math = not s.endswith("$$")
            out.append(ln); continue
        if in_math:
            if "$$" in s:
                in_math = False
            out.append(ln); continue
        if "id=\"refs\"" in ln:                  # reference list follows
            in_refs = True
            out.append(ln); continue
        if in_refs:
            out.append(ln); continue
        if not s:                                # blank line ends paragraph
            in_para = False
            out.append(ln); continue
        if ln.startswith(INDENT):                # already indented
            in_para = True
            out.append(ln); continue
        if s.startswith(("#", "|", "<", "---")):  # headings / tables / HTML
            out.append(ln); continue
        if not in_para:                          # first line of a paragraph
            out.append(INDENT + ln)
            in_para = True
            n += 1
        else:                                    # continuation line
            out.append(ln)
    path.write_text("\n".join(out), encoding="utf-8")
    return n


if __name__ == "__main__":
    indent = "--indent" in sys.argv
    files = [Path(p) for p in sys.argv[1:] if not p.startswith("--")] or DEFAULT
    for f in files:
        c = indent_paragraphs(f) if indent else run(f)
        tag = "paragraphs indented" if indent else "spaces inserted"
        print(f"{f.name}: {c} {tag}")
