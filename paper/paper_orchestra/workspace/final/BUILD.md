# DASFAA 2027 manuscript build

`paper.tex` and `paper.pdf` are the canonical anonymous submission source and PDF. The manuscript uses the official Springer LNCS class and `splncs04` bibliography style.

From this directory:

```powershell
..\..\tools\tectonic-0.17.0\tectonic.exe --keep-logs --keep-intermediates paper.tex
```

Run the complete project and submission gates in the required environment:

```powershell
D:\envs_dirs\no_1\python.exe -m pytest -q -p no:cacheprovider D:\ecomic\github\LexiRiskLabel\tests
```

Current verified state (2026-09-08): 16 pages, 188-word source abstract under the project test tokenizer, 76 tests passed, anonymous PDF, no unresolved citations/references, and no overfull boxes. Two nonfatal underfull-vbox warnings remain at natural section/page breaks. The manuscript integrates the five-policy IG block and the seed-controlled mechanism confirmation, matches the policy-specific candidate and budget implementation, and uses the quantity--risk trade-off as its unified narrative.
