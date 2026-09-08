# DASFAA manuscript package

The canonical anonymous submission is `workspace/final/paper.tex` with compiled
output `workspace/final/paper.pdf`. The main file imports LaTeX fragments from
`workspace/sections/`, figures from `workspace/figures/`, and verified BibTeX
entries from `workspace/literature/`.

Build from `workspace/final/` with the official Springer LNCS class:

```powershell
..\..\tools\tectonic-0.17.0\tectonic.exe --keep-logs --keep-intermediates paper.tex
```

The local Tectonic executable is not committed. Any compatible Tectonic or
LaTeX/BibTeX installation can build the same source. `FINAL_QA.md` records the
current page, abstract, compilation, visual, and test checks. The
`workspace/refinement/evidence_integrated_20260908/` directory contains the
latest PaperOrchestra refinement decision and rollback snapshots.
