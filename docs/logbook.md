# Logbook

Dated entries, newest first, ≤ 5 lines each: what was tried, the number, the verdict, and the
script or commit that measured it. (Older history is folded in from the previous docs during the
restructure's Phase G.)

## 2026-10-14 — restructure, Phases A–B

Tagged `pre-restructure`; golden outputs of the old code (`scripts/verify/dump_golden.py`,
`run_tables.sh`) are bitwise reproducible run to run, and every LLM replay is 100% cache hits.
Phase B moved the HSMM inference kernels into `hsmm/kernels.py`, added `class HSMM`
(`hsmm/model.py`), `class BadSets` (`detectors/thresholds.py`) and `core.py`, and shimmed the old
modules onto them. Equivalence and cost results: see `docs/restructure.md` "Progress notes".
