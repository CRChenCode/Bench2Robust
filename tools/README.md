# Scenario-control evidence: verification scripts

This directory holds integrity and consistency checks for the included traces in [`../traces/`](../traces/). They check the shipped records and their stated summaries; they do not regenerate the traces. All three need only Python 3.10+ and the standard library — no network, GPU, model or benchmark data.

- **`count_blocked_calls.py`** — for every trace listed in `appendix_spec.json` (B1-B4, C1-C4), counts the calls to tools in the episode's blocked set and how many returned data, were executed, or changed the environment state. Prints the totals for paths (b) and (c) and exits non-zero if any of them returned data.

  ```bash
  python3 -I tools/count_blocked_calls.py .
  ```

- **`render_appendix.py`** (with **`appendix_spec.json`**) — re-renders the Appendix A tables of the evidence document directly from the trace files, so the printed numbers can be checked against what the document claims.

  ```bash
  python3 -I tools/render_appendix.py tools/appendix_spec.json
  ```

- **`compact_trace.py`** — prints one trace file as a per-call table, for manual inspection.

  ```bash
  python3 -I tools/compact_trace.py traces/<file>.jsonl
  ```
