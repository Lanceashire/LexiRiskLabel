"""Submission-facing integrity gates for the DASFAA manuscript."""

from __future__ import annotations

import re
from pathlib import Path

import fitz


REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "paper" / "paper_orchestra" / "workspace" / "final"
MAIN = FINAL / "paper.tex"


def _expanded_tex(path: Path, seen: set[Path] | None = None) -> str:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        child = (path.parent / match.group(1)).with_suffix(".tex")
        assert child.is_file(), f"missing included TeX file: {child}"
        return _expanded_tex(child, seen)

    return re.sub(r"\\(?:input|include)\{([^}]+)\}", replace, text)


def test_official_lncs_and_abstract_limit() -> None:
    main = MAIN.read_text(encoding="utf-8")
    assert r"\documentclass[runningheads]{llncs}" in main
    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S
    )
    assert abstract
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract.group(1))
    assert len(words) <= 200, f"abstract has {len(words)} words"


def test_all_citations_and_cross_references_resolve() -> None:
    tex = _expanded_tex(MAIN)
    cite_keys: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        cite_keys.update(key.strip() for key in group.split(","))

    bib_names = re.search(r"\\bibliography\{([^}]+)\}", tex)
    assert bib_names
    bib_text = ""
    for name in bib_names.group(1).split(","):
        path = (FINAL / name.strip()).with_suffix(".bib")
        assert path.is_file(), f"missing bibliography: {path}"
        bib_text += path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
    assert not (cite_keys - bib_keys), f"missing citation keys: {cite_keys - bib_keys}"

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", tex))
    assert not (refs - labels), f"missing labels: {refs - labels}"


def test_figures_exist_and_stale_claims_are_absent() -> None:
    tex = _expanded_tex(MAIN)
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        if "#" in name:  # macro definition; concrete invocations are checked below
            continue
        assert (FINAL / name).resolve().is_file(), f"missing figure: {name}"
    for name in re.findall(r"\\paperfigure\{([^}]+)\}", tex):
        assert (FINAL / "../figures" / name).resolve().is_file(), f"missing figure: {name}"

    lowered = re.sub(r"\s+", " ", tex.lower())
    forbidden = (
        "shared candidate pool",
        "shared numerical budget",
        "matched count and risk",
        "matched quantity and risk",
        "quantity--quality frontier",
        "epsilon-greedy baseline",
        "chronologically ordered",
        "first-ever",
    )
    assert not [phrase for phrase in forbidden if phrase in lowered]


def test_compiled_pdf_is_anonymous_and_within_page_limit() -> None:
    pdf = FINAL / "paper.pdf"
    assert pdf.is_file()
    with fitz.open(pdf) as document:
        assert document.page_count <= 16
        text = "\n".join(page.get_text() for page in document)
    for token in ("Lanceashire", "LexiRiskLabel", "ecomic", "D:\\"):
        assert token.lower() not in text.lower(), f"identity/path leak: {token}"
