#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Add cross-reference hyperlinks to an existing Word paper (in place).

Works on the CURRENT docx (preserving any manual edits made in Word):
- bookmarks on figure captions (figN), table captions (tabN), numbered
  display equations (eqN) and reference entries (refN)
- internal links: in-text 图N / 表N / 式(N) mentions, superscript citations
  [n] / [n-m] / [n,m] (per-number links)
- external links for http(s) URLs in the reference list
Links keep black text (no Hyperlink style), per journal print convention;
Ctrl+click in Word/WPS jumps.

Usage: python add_crosslinks.py [docx_path]
The original file is copied to <name>.bak before saving.
"""
import copy
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor
from docx.text.run import Run

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "manuscript_cn.docx"
BAK = SRC.with_suffix(".docx.bak")

_cite_pat = re.compile(r"^\[\d+(?:[-,，]\d+)*\]$")
_num_pat = re.compile(r"\d+")
_sep_pat = re.compile(r"[-,，]")
_mention_pat = re.compile(r"图(\d+)|表(\d+)|式[(（](\d+)[)）]")
_url_pat = re.compile(r"https?://[A-Za-z0-9\-._~:/?#@!$&'()*+,;=%\[\]]+")

_bm_id = iter(range(100, 100000))


def get_text(r_el):
    return "".join(t.text or "" for t in r_el.findall(qn("w:t")))


def set_run_text(r_el, text):
    ts = r_el.findall(qn("w:t"))
    if not ts:
        t = OxmlElement("w:t")
        r_el.append(t)
        ts = [t]
    ts[0].text = text
    ts[0].set(qn("xml:space"), "preserve")
    for extra in ts[1:]:
        r_el.remove(extra)


def clone_run(r_el, text):
    nr = copy.deepcopy(r_el)
    set_run_text(nr, text)
    return nr


def wrap_anchor(r_el, anchor=None, rid=None):
    hl = OxmlElement("w:hyperlink")
    if anchor:
        hl.set(qn("w:anchor"), anchor)
    if rid:
        hl.set(qn("r:id"), rid)
    hl.set(qn("w:history"), "1")
    hl.append(r_el)
    return hl


def add_bookmark(p, name, existing):
    """Add a bookmark unless one with this name already exists (rerun-safe)."""
    if name in existing:
        return False
    sid = str(next(_bm_id))
    st = OxmlElement("w:bookmarkStart")
    st.set(qn("w:id"), sid)
    st.set(qn("w:name"), name)
    en = OxmlElement("w:bookmarkEnd")
    en.set(qn("w:id"), sid)
    pPr = p._element.find(qn("w:pPr"))
    if pPr is not None:
        pPr.addnext(st)
    else:
        p._element.insert(0, st)
    p._element.append(en)
    existing.add(name)
    return True


def replace_run(p, r_el, new_els):
    for el in new_els:
        r_el.addprevious(el)
    r_el.getparent().remove(r_el)


def link_citation_group(p, run_els, txt, ref_bm, rep):
    nums = _num_pat.findall(txt)
    seps = _sep_pat.findall(txt)
    tpl = run_els[0]
    new = [clone_run(tpl, "[")]
    for k, n in enumerate(nums):
        if k:
            new.append(clone_run(tpl, seps[k - 1]))
        nr = clone_run(tpl, n)
        if n in ref_bm:
            new.append(wrap_anchor(nr, anchor=f"ref{n}"))
            rep["cite_links"] += 1
        else:
            new.append(nr)
            rep["cite_missing"].append(n)
    new.append(clone_run(tpl, "]"))
    replace_run(p, run_els[0], new)
    for r_el in run_els[1:]:
        r_el.getparent().remove(r_el)


def linkify_mentions_crossrun(p, bm, rep):
    """Cross-run mention linking: Word splits runs at edit-session (rsid)
    boundaries, so 图N/表N/式(N) often spans several runs. Concatenate the
    paragraph's direct runs, locate matches, cut runs apart and wrap each
    match in an internal hyperlink."""
    runs = [el for el in p._element if el.tag == qn("w:r")]
    texts = [get_text(r) for r in runs]
    full = "".join(texts)
    matches = list(_mention_pat.finditer(full))
    if not matches:
        return

    def refresh():
        nonlocal runs, texts, full
        runs = [el for el in p._element if el.tag == qn("w:r")]
        texts = [get_text(r) for r in runs]
        full = "".join(texts)

    # process right-to-left so earlier offsets stay valid
    for m in reversed(matches):
        kind = "fig" if m.group(1) else ("tab" if m.group(2) else "eq")
        n = m.group(1) or m.group(2) or m.group(3)
        if n not in bm[kind]:
            rep["mention_missing"].append(f"{kind}{n}")
            continue
        anchor = f"{kind}{n}"
        rep["mention_links"] += 1

        starts = []
        off = 0
        for t in texts:
            starts.append(off)
            off += len(t)
        ms, me = m.start(), m.end()
        ri_s = max(k for k in range(len(runs)) if starts[k] <= ms)
        ri_e = max(k for k in range(len(runs)) if starts[k] < me)
        before = full[starts[ri_s]:ms]
        match_txt = full[ms:me]
        after = full[me:starts[ri_e] + len(texts[ri_e])]

        new_els = []
        if before:
            new_els.append(clone_run(runs[ri_s], before))
        new_els.append(wrap_anchor(clone_run(runs[ri_s], match_txt), anchor=anchor))
        if after:
            new_els.append(clone_run(runs[ri_e], after))
        for el in new_els:
            runs[ri_s].addprevious(el)
        for k in range(ri_s, ri_e + 1):
            runs[k].getparent().remove(runs[k])
        refresh()


def link_urls_in_run(p, r_el, rep):
    text = get_text(r_el)
    matches = list(_url_pat.finditer(text))
    if not matches:
        return
    pieces = []
    pos = 0
    for m in matches:
        url = m.group(0).rstrip(".")
        end = m.start() + len(url)
        pieces.append((text[pos:m.start()], None))
        pieces.append((url, url))
        pos = end
    pieces.append((text[pos:], None))
    new = []
    for t, a in pieces:
        if not t:
            continue
        nr = clone_run(r_el, t)
        if a:
            rid = p.part.relate_to(a, RT.HYPERLINK, is_external=True)
            new.append(wrap_anchor(nr, rid=rid))
            rep["url_links"] += 1
        else:
            new.append(nr)
    replace_run(p, r_el, new)


def is_sup(r_el):
    rPr = r_el.find(qn("w:rPr"))
    if rPr is None:
        return False
    va = rPr.find(qn("w:vertAlign"))
    return va is not None and va.get(qn("w:val")) == "superscript"


LINK_BLUE = RGBColor(0x05, 0x63, 0xC1)  # Word standard hyperlink blue


def colorize_links(doc):
    """Blue (no underline) for all hyperlink runs — journal-friendly look."""
    n = 0
    for hl in doc.element.body.iter(qn("w:hyperlink")):
        for r_el in hl.findall(qn("w:r")):
            Run(r_el, None).font.color.rgb = LINK_BLUE
            n += 1
    return n


def main():
    doc = Document(SRC)
    paras = doc.paragraphs
    rep = {"cite_links": 0, "mention_links": 0, "url_links": 0,
           "cite_missing": [], "mention_missing": [], "split_miss": 0}

    # existing bookmark names (avoid collisions)
    existing = {bs.get(qn("w:name")) for bs in doc.element.body.iter(qn("w:bookmarkStart"))}

    # refs section start
    ref_idx = next((i for i, p in enumerate(paras) if p.text.strip() == "参考文献"), None)
    assert ref_idx is not None, "参考文献 heading not found"

    # ---- bookmarks on targets (deduped -> rerun-safe)
    bm = {"fig": {}, "tab": {}, "eq": {}, "ref": {}}
    n_cont = 0
    n_bm_new = 0
    for p in paras:
        style = p.style.name
        t = p.text.strip()
        if style == "Image Caption":
            m = re.match(r"^图(\d+)", t)
            if m:
                n_bm_new += add_bookmark(p, f"fig{m.group(1)}", existing)
                bm["fig"][m.group(1)] = True
        elif style == "Table Caption":
            m = re.match(r"^表(\d+)", t)
            if m:
                n_bm_new += add_bookmark(p, f"tab{m.group(1)}", existing)
                bm["tab"][m.group(1)] = True
        elif p._element.find(qn("m:oMath")) is not None:
            # tolerate user-typed punctuation before the number, e.g. "\t.\t(2)"
            m = re.search(r"\((\d+)\)\s*$", t)
            if m:
                n_bm_new += add_bookmark(p, f"eq{m.group(1)}", existing)
                bm["eq"][m.group(1)] = True
            elif t == "":
                n_cont += 1
    for p in paras[ref_idx + 1:]:
        m = re.match(r"^\[(\d+)\]", p.text.strip())
        if m:
            n_bm_new += add_bookmark(p, f"ref{m.group(1)}", existing)
            bm["ref"][m.group(1)] = True

    # ---- citations (superscript runs) -> per-number internal links
    for p in paras[:ref_idx]:
        runs = [r._element for r in p.runs]
        i = 0
        while i < len(runs):
            if not is_sup(runs[i]):
                i += 1
                continue
            j = i
            while j + 1 < len(runs) and is_sup(runs[j + 1]):
                j += 1
            grp = runs[i:j + 1]
            txt = "".join(get_text(r) for r in grp).strip()
            if _cite_pat.match(txt):
                link_citation_group(p, grp, txt, bm["ref"], rep)
            else:
                done = False
                for r in grp:
                    t1 = get_text(r).strip()
                    if _cite_pat.match(t1):
                        link_citation_group(p, [r], t1, bm["ref"], rep)
                        done = True
                if not done:
                    rep["split_miss"] += 1
            i = j + 1

    # ---- in-text mentions 图N/表N/式(N) -> internal links (skip captions)
    # cross-run: Word splits runs at edit-session boundaries, so the mention
    # text often spans several runs
    for p in paras[:ref_idx]:
        if p.style.name in ("Image Caption", "Table Caption"):
            continue
        linkify_mentions_crossrun(p, bm, rep)

    # ---- external URLs in the reference list
    for p in paras[ref_idx + 1:]:
        for r_el in list(p._element):
            if r_el.tag == qn("w:r"):
                link_urls_in_run(p, r_el, rep)

    n_colored = colorize_links(doc)

    shutil.copy2(SRC, BAK)
    doc.save(SRC)

    print(f"OK -> {SRC}")
    print(f"backup: {BAK}")
    print(f"bookmarks: new={n_bm_new} | targets available: fig={len(bm['fig'])} "
          f"tab={len(bm['tab'])} eq={len(bm['eq'])} ref={len(bm['ref'])} "
          f"(eq continuation lines: {n_cont})")
    print(f"internal links: citations={rep['cite_links']} "
          f"mentions(fig/tab/eq)={rep['mention_links']} | external URLs={rep['url_links']} "
          f"| colored runs={n_colored}")
    if rep["cite_missing"]:
        print(f"WARN missing ref targets for citations: {sorted(set(rep['cite_missing']))}")
    if rep["mention_missing"]:
        print(f"WARN missing targets for mentions: {sorted(set(rep['mention_missing']))}")
    if rep["split_miss"]:
        print(f"WARN superscript groups not recognized as citations: {rep['split_miss']}")


if __name__ == "__main__":
    main()
