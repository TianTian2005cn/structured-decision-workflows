# Structured Decision Models in Multi-Tool Agent Workflows — replication code

Code, frozen protocols and per-item results for the paper
**"Evaluating Structured Decision Models in Multi-Tool Agent Workflows: Quality, Cost, and Execution Reliability"**
(CAiSE 2027 submission).

The repository lets you (a) recompute every reported number **offline** from the stored results and exact
request/response caches, and (b) re-run the experiments live with your own API keys.

## Layout

| Path | Content | Paper |
|---|---|---|
| `jevset/` | SDM client with exact-request cache (no headers stored), frozen decision pipeline (`v2.py`, `algorithms.py`: top-k, 24-candidate portfolio, constraint-aware MAP decoder, routing policy .65/.15, one-step recovery), closed workflow domains and independent oracles (`workflow_v2.py` ledger reports, `workflow_order.py` order invoicing, `workflow_scale.py` 400-workflow development strata) | Sect. 3–4 |
| `scripts/run_sealed_confirmation.py` | Pre-registered one-shot test, domain 1 (verifies frozen source hashes and the protocol before any request) | Tables 1–2 |
| `scripts/run_order_domain.py` | Pre-registered one-shot test, domain 2 | Tables 1–2 |
| `scripts/run_sealed_replication.py`, `run_sealed_run{3,4,5}.py` | Independent runs with fresh request caches | Sect. 5.1 |
| `scripts/run_sealed_nl_variant.py` | Natural-language request variant | Fig. 2, Table 1 |
| `scripts/run_frontier_closed.py` | Frontier planners (DeepSeek V4.1 Flash, V4 Pro) and Flash token-probability scorer with the identical decoder | Table 1 |
| `scripts/analyze_*.py`, `summ_frontier.py` | Exact McNemar tests with Holm correction, Wilson intervals, run-to-run stability | Tables 1–2 |
| `scripts/rho_sensitivity.py` | Offloadable cost share ρ at three constructor levels, trajectory bootstrap | Table 4 |
| `scripts/run_workflow_scale.py`, `run_v2_development.py`, `run_live_pilot.py` | Development studies (400 workflows) and shared helpers | Fig. 2 (left) |
| `baselines/` | Lexical scorer, open-weights (Qwen2.5-7B, llama.cpp) scorer and planner, NL templates | supplementary |
| `figures/` | Figure and RQ2-table scripts (forest data, MDE, Holm, random-effects pooling) | Figs. 2–3, Table 3 |
| `outputs/paper03_caise/` | Frozen protocols, per-item results, request/response caches, usage, analyses of all new experiments | Sect. 5 |
| `outputs/direct_v3/actor_states.json` | Matched executor calls of the cost audit (4,040 calls, 568 trajectories) | Table 4 |
| `outputs/v3/workflow_scale/` | 400-workflow development study: protocol, analysis, report | Fig. 2 |
| `data/`, `results_private/v2_sealed/` | Development workflows and the held-out test sets (domain 1 workflows, MetaTool items) | Sect. 4 |

## Requirements

Python 3.10+. The decision pipeline, environments and analyses use only the standard library.
Optional: `matplotlib` (figures), `numpy` + `llama-cpp-python` (open-weights baselines).

```bash
pip install -r requirements.txt   # optional extras only
```

## Reproduce offline (no API key needed)

Exact request caches are stored next to each experiment, so re-running a script with the same inputs reads
the cached responses instead of calling an API.

```bash
python scripts/analyze_sealed_confirmation.py   # domain 1: pre-registered tests H1-H4, Wilson CIs
python scripts/analyze_order_domain.py          # domain 2
python scripts/analyze_sealed_runs.py           # five-run stability
python scripts/analyze_sealed_metatool.py       # MetaTool held-out set
python scripts/summ_frontier.py                 # frontier planners / scorer swap
python scripts/rho_sensitivity.py               # offloadable cost share (Table 4)
cd figures && python rq2_table.py && python meta_rq2.py && python figs_closed.py && python figs_open.py
```

## Re-run live

Copy `.env.example` to `.env` and fill in `TYPESAFE_API_KEY` (structured decision model, `jev-1.14.0`) and/or
`DEEPSEEK_API_KEY`. Use a **new** output directory for an independent replication: the SDM is not
bit-deterministic, and cache hits are not independent model calls.

```bash
python scripts/run_sealed_confirmation.py workflow metatool
python scripts/run_order_domain.py
python scripts/run_frontier_closed.py e1 deepseek-flash ledger
```

Open-weights baselines: set `QWEN_GGUF` to a local `Qwen2.5-7B-Instruct-Q4_K_M.gguf`.

## Notes

- Credentials are never written to caches or logs.
- The public-benchmark studies (MetaTool, ToolSandbox, ToolMaze, CCTU, τ-bench) depend on those benchmarks'
  own repositories and licences; their aggregate results are reported in the paper.
- Prices used for cost estimates were the providers' public list prices at the time of the experiments.
