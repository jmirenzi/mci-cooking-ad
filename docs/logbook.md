# Logbook

Dated entries, newest first, ≤ 5 lines each: what was tried, the number, the verdict, and the
script or commit that measured it. (Older history is folded in from the previous docs during the
restructure's Phase G.)

## 2026-10-14 — restructure, Phase C

`LiveHSMMDetector`, `LagHSMMDetector`, `HindsightHSMMDetector` and `narrate` (detectors/)
reproduce every golden bitwise at 1.00× cost. π̃-from-the-filter (unification 1) flips flags on
1/582 Breakfast trials and was rejected; r̂-from-the-filter (2) agrees 582/582 but costs too much.
Details: `docs/restructure.md` "Progress notes".

## 2026-10-14 — restructure, Phases A–B

Tagged `pre-restructure`; golden outputs of the old code (`scripts/verify/dump_golden.py`,
`run_tables.sh`) are bitwise reproducible run to run, and every LLM replay is 100% cache hits.
Phase B moved the HSMM inference kernels into `hsmm/kernels.py`, added `class HSMM`
(`hsmm/model.py`), `class BadSets` (`detectors/thresholds.py`) and `core.py`, and shimmed the old
modules onto them. Equivalence and cost results: see `docs/restructure.md` "Progress notes".
