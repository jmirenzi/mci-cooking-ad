# Working in this repo

Start at `docs/README.md` (the system, the classes, how to run it). History is in
`docs/logbook.md`.

- **Update `docs/` in the same commit as any code it describes.** Docs explain the current theory and
  where it lives in the code (each has a "Code" table of `file:class.method`). Past attempts and
  measurements go to `docs/logbook.md` as dated entries, newest first, ≤ 5 lines each.
- **Every results table states its metric letter** (G is the headline; E only for comparing with
  PREGO; see `docs/eval.md`). A number without a letter is not citable.
- **Run Python through `./py`** (`./py -m pytest -q`, `./py scripts/...`). It pins `PYTHONPATH` to this
  checkout and stops JAX preallocating the GPU.
- **Never `uv sync` without `--extra dev --extra gpu`.** A bare sync uninstalls the CUDA jaxlib and JAX
  silently falls back to CPU.
- **LLM replays need the original `--base-url` and `--max-requests 0`.** The response cache key includes
  the base URL: Breakfast was collected through `http://localhost:11435/v1`, EPIC-Tent through
  `http://localhost:11434/v1`. With `--max-requests 0` a miss raises instead of sending a request.
  Never delete `dataset/processed/*/llm_cache/`.
- **Never `pkill -f` / `pgrep -f` a pattern that matches your own shell** (e.g. the script name you
  launched it with). Stop a process by PID, or match the interpreter's executable.
- **Say "run", not "step"**: a run is a maximal stretch of constant (verb, noun). Users' steps are not
  observed.
