# Bench2Robust

Bench2Robust is a controllable robustness harness for evaluating tool-using agents when APIs are unreliable. It injects deterministic, configurable failures into otherwise clean benchmark environments while preserving constraints such as write-action safety, injection budgets, and bounded consecutive failures.

> **Anonymous review status:** this repository is an anonymized snapshot prepared for double-blind peer review. It is not actively maintained during review.

> This code is being shared solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated anonymous submission. Pull requests are not being accepted during review in order to maintain the artifact exactly as it was used in the paper.

## What is included

| Path | Contents | Runs on its own? |
|---|---|---|
| `bench2robust/` | The library: nine explicit and silent uncertainty types, the deterministic injection policy and response cache, the benchmark-agnostic injection engine, the scenario controller (S1/S2/S3), and adapters/runner for τ²-bench, VitaBench and BFCL. This is the corrected version (v2) — it fixes the bug identified during review. | yes (standard library only; τ²-bench optional) |
| `tests/` | Library tests (network-free). | yes |
| `traces/` | Scripted blocked-path traces (JSONL), one record per tool call. | n/a |
| `tools/` | Verification scripts a reviewer can run against the shipped traces: `count_blocked_calls.py`, the appendix renderer (`render_appendix.py` + `appendix_spec.json`), and a trace viewer (`compact_trace.py`). See `tools/README.md`. | yes |
| `eval/src/run_inference.py` | The original (pre-fix) evaluation harness behind the reported results. Anonymised in place with line numbers unchanged. | no (for inspection) |
| `training/bench2robust_rl/rollout.py`, `training/slime/LICENSE` | The original (pre-fix) RL training environment behind the reported results, plus the upstream slime licence. Same anonymisation. | no (for inspection) |

Raw trajectories, model outputs, datasets, evaluation logs, checkpoints, and the rest of the modified training/evaluation framework fork. These are large research artifacts with separate provenance and redistribution requirements, and none is needed to answer the reviewer's question.

`eval/` and `training/` (the original, pre-fix code) are anonymised in place: placeholders mark local paths and deployment settings (`<PATH>`, `<CREDS_FILE>`, `<IMDS_ENDPOINT>`, `<REGION>` and similar); no line was added or removed. `bench2robust/` has no such paths to begin with; its files carry only the standard `Copyright (c) 2026 Anonymous Authors` header.

## Requirements

- Python 3.10 or newer for the dependency-free core library
- Python 3.12 or 3.13 for the optional τ²-bench integration (installed from the upstream repository, see below)
- Optional benchmark dependencies only when using their runner or adapter

## Install

Install the core library:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On systems where `python3 -m venv` fails because `ensurepip` is missing (for example Debian/Ubuntu without `python3-venv`), create the environment without pip and bootstrap pip from the system pip:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install pip
source .venv/bin/activate
python -m pip install -e .
```

Install the τ²-bench integration (Python 3.12 or 3.13; it is fetched from the upstream GitHub repository, so it needs network access and `git`):

```bash
python -m pip install -e ".[tau2]"
```

No credentials are needed to import or test the core package.

## Minimal example

```python
from bench2robust import EpisodeUncertaintyConfig, GenericInjectionEngine, ToolConfig

config = EpisodeUncertaintyConfig(
    tool_configs={
        "lookup": ToolConfig(
            name="lookup",
            is_write_action=False,
            has_alternatives=False,
            uncertainty_priors={"clean": 0.5, "timeout": 0.5},
        )
    },
    max_consecutive_failures=1,
    max_total_injections=2,
    seed=42,
)

engine = GenericInjectionEngine(config)
should_inject, noise_type = engine.should_inject("lookup")
clean_response = {"status": "ok", "value": 7}

if should_inject:
    observed_response = engine.inject(noise_type, "lookup", clean_response)
else:
    engine.record_clean("lookup", clean_response)
    observed_response = clean_response

print(observed_response)
```

## Noise model

| Category | Type | Typical recovery |
| --- | --- | --- |
| Explicit | Timeout | Retry with a bounded policy |
| Explicit | Rate limit | Wait, then retry |
| Explicit | Server error | Retry with a bounded policy |
| Explicit | Authentication error | Refresh credentials or escalate |
| Explicit | Malformed response | Retry or report serialization failure |
| Explicit | Schema drift | Validate and escalate incompatible changes |
| Silent | Partial response | Cross-check decision-critical fields |
| Silent | Stale response | Compare against another source or timestamp |
| Silent | Factual error | Cross-check decision-critical values |

Injection is disabled for configured write actions and selected meta-tools. A seed controls sampling so experiments can be repeated.

Scenario control (S1/S2/S3) is separate from stochastic injection (`bench2robust/uncertainty/scenario.py`). Scenario-blocked tools are never executed and stay blocked until episode reset, regardless of B and K_max; unblocked fallback tools are exempt from stochastic noise (configurable via `fallback_noise`); all other tools follow the stochastic policy.

In detail, each tool call first goes through the episode's `ScenarioController`. A call to a scenario-blocked tool returns an explicit blocked response, is never executed, never consults the stochastic policy (`can_inject` / `should_inject`), never consumes the injection budget, consecutive-failure or per-error counters, and never receives a silent corruption. Blocking persists until `reset()` installs the next episode's scenario. A call to an unblocked fallback tool (`ScenarioSpec.fallback_paths`: the unblocked side of an S2 equivalence class; derived from the domain's equivalence classes when only a blocked tool set is given) returns a `CLEAN` decision without consulting the stochastic policy, so it never receives an explicit failure or a silent corruption and never consumes the budget or consecutive-failure counters. `fallback_noise` selects this behaviour: `"exempt"` (default), `"exempt_s1"` (additionally, in S1 one randomly chosen side of each equivalence class is noise-free), or `"noisy"` (fallback tools follow the stochastic policy). It is accepted by `ScenarioController`, `GenericInjectionEngine`, the VitaBench and BFCL adapters, and `Tau2BenchRunner.run_episode`. In S1 (no blocking) and S3 (no unblocked side) no tool is exempt under the default. `GenericInjectionEngine.decide()` returns an explicit `Decision` (`SCENARIO_BLOCK`, `STOCHASTIC_EXEC_FAILURE`, `STOCHASTIC_SILENT_CORRUPTION`, `CLEAN`) that tells the adapter whether the tool may be executed; `Decision.fallback_exempt` marks the clean decisions of exempt fallback tools.

## Optional benchmark integrations

Adapters are imported lazily. Install each benchmark from its upstream project and follow that project's license and setup instructions before using its integration.

`TauBenchAdapter` exposes domain tool classifications. Episode execution and environment interception are owned by `Tau2BenchRunner`, avoiding a second wrapper with divergent behavior:

```python
from bench2robust import get_adapter
from bench2robust.runners import Tau2BenchRunner

metadata = get_adapter("tau_bench", domain="retail")
print(metadata.get_tool_list())

runner = Tau2BenchRunner()
domain_config = runner.get_domain_config("retail")
print(sorted(domain_config.write_tools))
```

The optional `tau2` extra installs τ²-bench (package `tau2`, version 1.0.0) from `github.com/sierra-research/tau2-bench` at commit `5ebebbe8`, the version used for the experiments. Do not use `pip install tau2` from PyPI: the PyPI project of that name is an unrelated package. The released supplement pinned `tau2==1.0.1`, which does not exist on PyPI. Set `TAU2_DATA_DIR` to the `data/` directory of a τ²-bench checkout if the installed package cannot find its data. For an explicitly selected local τ² source checkout, set `BENCH2ROBUST_TAU2_PATH` to its `src` directory. Set `BENCH2ROBUST_NL_EVAL_MODEL` only when intentionally overriding τ²-bench's default natural-language evaluator model.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q bench2robust tests
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The core test suite is deterministic and does not require a network, GPU, model endpoint, cloud account, or benchmark dataset.

## Verification for reviewers

This section explains which code produced the scenario-controlled results, how that code relates to the library supplied with the submission, and the blocked-path traces. Everything below needs only Python 3.10+ and the standard library; nothing needs a network, GPU, model or benchmark data.

**1. Check file integrity.**

```bash
sha256sum -c MANIFEST.sha256
```

**2. Run the tests** (36 tests; the τ²-bench runner variants run as additional subtests only if τ²-bench is installed via the `tau2` extra: 77 subtests without it, 91 with it).

```bash
python -m pip install pytest
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests
```

The README's `unittest` command under [Test](#test) runs the same tests without pytest.

**3. Check the blocked-call count in the included traces (one command).**

```bash
python3 -I tools/count_blocked_calls.py .
```

It reads the trace files listed in `tools/appendix_spec.json` and prints, per trace, the number of calls to blocked tools, how many returned data (a response that is not an explicit error), how many executed the tool, and how many changed the environment state. Expected result: on paths (b) and (c) — the original evaluation harness and the original training environment — every blocked call in the traces returned an explicit error, in both S2 and S3: 102 blocked calls (S2 48, S3 54), 0 of which returned data. The script exits non-zero if any blocked call on path (b) or (c) returned data. `python3 -I tools/render_appendix.py tools/appendix_spec.json` re-renders Appendix A of the evidence document from the same files.

**4. Read the original source behind traces B and C.** [`eval/src/run_inference.py`](eval/src/run_inference.py) is the original evaluation harness, and [`training/bench2robust_rl/rollout.py`](training/bench2robust_rl/rollout.py) is the original RL training environment. Both are anonymised in place, so line numbers are unchanged. Episode-persistent blocking is at `run_inference.py:367-388` and `rollout.py:982-994`. These files are for inspection only and do not run on their own. `rollout.py` sits in a fork of the Apache-2.0 slime framework (licence in `training/slime/LICENSE`).

The two experimental paths enforce the same persistent agent-visible block but differ in execution order. The evaluation harness uses **pre-execution suppression**: it checks the blocked set and returns an explicit error without executing the tool. The training environment uses **execute-then-mask**: it executes the tool, then checks the blocked set and replaces the observation with an explicit error. Both paths handle the blocked set before consulting the stochastic policy, so blocking cannot lapse when B or K_max is exhausted and blocked calls do not consume those stochastic-policy counters. All tools eligible for scenario blocking in the reported experiments are read-only. Consequently, for the training traces, execute-then-mask is equivalent at the agent-observation and environment-state levels: the agent never receives the underlying tool result, and every blocked call has an unchanged environment-state hash. This ordering distinction is nevertheless reported explicitly because only the evaluation path is strictly execution-suppressing.

**What each trace file shows.** All traces use τ²-bench Retail task 66 (synthetic benchmark data), B=1 and K_max=1. "B, K_max used up" means a stochastic injection on an unblocked tool exhausts both limits before the blocked tool is called again.

| File (under `traces/`) | Trace | Code path | Scenario | What it shows |
|---|---|---|---|---|
| `eval_retail_s2.jsonl`, `eval_retail_s3.jsonl` | B1, B2 | (b) original evaluation harness | S2, S3 | Repeated calls to the same blocked tool (up to 5 in a row, beyond K_max) all return errors; blocked tools are not executed; policy counters untouched. |
| `eval_counter_separation.jsonl`, `eval_s3_counter_separation.jsonl` | B3, B4 | (b) original evaluation harness | S2, S3 | Blocked tools stay blocked after B and K_max are used up. |
| `retail_s2.jsonl`, `retail_s3.jsonl` | C1, C2 | (c) original training environment | S2, S3 | As B1/B2, but the blocked tool is executed before its response is replaced (read-only tools; state hash unchanged). |
| `train_counter_separation.jsonl`, `train_s3_counter_separation.jsonl` | C3, C4 | (c) original training environment | S2, S3 | As B3/B4, with the same execute-then-replace behaviour. |

The corrected library's correctness (no lapse, fallback exemption, reset, etc.) is not re-demonstrated by trace in this release. `bench2robust/` is the corrected library itself, and `tests/test_core.py` / `tests/test_scenario_control.py` check it directly with `python -m unittest discover -s tests` (see [Test](#test) above).

**How the traces were generated.** Each trace replayed a fixed call plan on τ²-bench Retail task 66 against the real evaluation `get_response` or the real training episode coroutine (model, user simulator and evaluator replaced by scripted stand-ins), or against the real library engine for the library traces. No LLM or GPU was used. The harness that generated them required the original/corrected code trees and τ²-bench/VitaBench sources, none of which is included, so it is not part of this release — `tools/` only ships what a reviewer can actually run (see `tools/README.md`): `count_blocked_calls.py`, `render_appendix.py` and `compact_trace.py`.

**Scope.** The traces for paths (b) and (c) were produced from the original experimental code tree. The files they exercise are included under `eval/` and `training/`. These are **included traces with integrity and consistency checks**, not independently regenerable traces: `MANIFEST.sha256` checks the shipped bytes, while `count_blocked_calls.py` and `render_appendix.py` check that the trace records support the stated counts, counter separation, execution-order distinction, and unchanged-state claims. The generation harness is not included, as described above.

## Data and reproducibility: building the scenario-controlled benchmark from public data

No task data are shipped. The scenario-controlled (S1/S2/S3) and injected versions are built at run time from the public benchmark data.

**1. Get τ²-bench and its data.** Install the `tau2` extra (see [Install](#install)), then clone the same upstream commit to get the data, which the pip package does not include:

```bash
git clone https://github.com/sierra-research/tau2-bench <TAU2_DIR>
git -C <TAU2_DIR> checkout 5ebebbe827b455b3ed04fcb9294235c6ef4e5fd6
export TAU2_DATA_DIR=<TAU2_DIR>/data
```

**2. Choose the task set.** The original evaluation harness loads tasks through τ²-bench's own loader (`get_tasks(<domain>, task_split_name=...)`, `--task-split`; see `eval/src/run_inference.py`). With the public data, the available Retail splits are those in `<TAU2_DIR>/data/tau2/domains/retail/split_tasks.json` (`base` 114, `train` 74, `test` 40).

**3. Make a held-out split** if you need one. The paper's train/held-out split was a single global shuffle of the task IDs:

```python
import json, random
ids = [t["id"] for t in json.load(open("tasks.json"))]
random.Random(42).shuffle(ids)
train, heldout = ids[:937], ids[937:]   # 937/402 for the 1,339-task Combined Retail set
```

For another task set, use the same procedure with your own cut point.

**4. Assign scenarios per task**, fixed across models. A `{task_id: {seed: S1|S2|S3}}` file is drawn from an md5 hash of (seed, task id); the helper that builds this file is part of the broader evaluation tooling and is **not included in this release** (out of scope for the scenario-blocking question this package addresses). For a fixed scenario instead, use the library's `ScenarioSpec` / `build_scenario_spec` (`bench2robust/uncertainty/scenario.py`) directly.

**5. Scenario blocking and injection.**
- The scenario's blocked set comes from the domain's tool equivalence classes. In the library these are `bench2robust/uncertainty/scenario.py:194` (`DOMAIN_EQUIVALENCE_CLASSES`) and `:282` (`build_scenario_spec`). The original evaluation harness has its own, equivalent tool pairs (`eval/src/run_inference.py`).
- Blocking is enforced by `ScenarioController.decide` / `blocked_response` (`scenario.py:435`, `:452`) before the stochastic policy (`bench2robust/uncertainty/injection_policy.py:167`, `decide`) is consulted.
- Per-task seeds come from `stable_task_seed` (`scenario.py:378`).

**6. Training data.** The slime prompt-file builder (`prepare_data.py`) is part of the broader RL training pipeline and is **not included in this release**, for the same reason as step 4.

**What the public data do not provide** (no code change can supply these, and is unrelated to the scenario-blocking question):
- **The alternative ("w/ alt") tools.** The experiments added these to τ²-bench's domain toolkits:
  - Retail: `get_user_by_order`, `get_orders_for_user`, `search_product_by_name`, `get_item_by_product_and_options`, `check_order_item_availability`;
  - Airline: `find_user_by_email`, `find_user_by_name_dob`, `get_reservations_for_user`, `search_flights`;
  - Telecom: further alternatives.

  They are not in upstream τ²-bench and not in this repository. Without them, "Inject w/ alt" S2 has no fallback path for two of the three Retail equivalence classes, and S3 blocks tools that do not exist. S1, and the "Inject w/o alt" condition, run on upstream τ²-bench as is.
- **The augmented task sets** Retail-3I (general/ambiguous/changing, 1,225 tasks) and Airline-3I, and therefore the 1,339-task Combined Retail set and its 937/402 split, are not public and are not part of this release.

This repository is sufficient to:
- exercise the uncertainty engine and scenario controller;
- recount the traces, which use the public base Retail task 66;
- inspect the code behind the reported results (the reviewer's question).

It is not sufficient to reproduce the paper's tables, nor the full evaluation/training pipeline — that is intentionally out of scope for this release.

## Security

Do not commit credentials, local cloud profiles, raw service logs, model traces, or generated conversations. Follow [`SECURITY.md`](SECURITY.md) for private vulnerability reporting.

## Citation and license

If you use this code, please cite the accompanying anonymous submission, "Bench2Robust: Scenario-Controlled Recovery for Tool-Using Agents".

This project is released under the Creative Commons Attribution-NonCommercial 4.0 International license (`CC-BY-NC-4.0`); the full text is in [`LICENSE`](LICENSE). `training/bench2robust_rl/rollout.py` is the authors' own code living inside a fork of the open-source [slime](https://github.com/THUDM/slime) RL framework, licensed under the Apache License 2.0 (text in `training/slime/LICENSE`); the CC-BY-NC-4.0 licence above covers the authors' own contributions and does not override the Apache-2.0 licence of slime or of any slime-derived material.
