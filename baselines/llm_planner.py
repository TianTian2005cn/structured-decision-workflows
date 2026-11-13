"""Generative LLM planner baseline for the controlled V2 workflow environment (paper 3, CAiSE 2027).

The planner receives EXACTLY the public state that the structured decision model (Jev) receives
(Sample.public_state(): instruction+request+ledger text, tool registry with meanings, declared
requires_any / precedence / exclusion constraints, costs and budget) and returns a tool set through
JSON-schema-constrained decoding (tool ids are an enum), so every output is well-typed.

Arms (all share the first call, cached by exact prompt hash):
  llm_static            one call, executed once
  llm_recovery          + the SAME one-initial-read recovery protocol used for the SDM arms
  llm_repair_recovery   + one validator-feedback repair if the set violates declared constraints/budget,
                          then the same recovery protocol
Greedy decoding (temperature 0), fixed seed. Prompt frozen before any sealed item is run.
Usage: python llm_planner.py <split: dev|sealed> <model.gguf> <outdir> [limit]
"""
from __future__ import annotations
import json, sys, time, hashlib
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevset.schema import Sample
from jevset.workflow_v2 import WorkflowTask, execute_workflow
from jevset.evaluation import topological_plan

SYSTEM = ("You are the planning component of a workflow engine. Given a task, a registry of tools and the "
          "declared constraints, select the exact SET of tools whose execution (in dependency order) produces "
          "the requested result. Respect every constraint: requires_any (a selected tool needs at least one of "
          "the listed tools), excludes (never select both), precedes_if_both_selected, and the total cost must "
          "not exceed the budget. Do not select tools that are unnecessary or forbidden by the instructions. "
          "Answer only with the JSON object.")

def user_prompt(state: dict) -> str:
    # Same information as the SDM public state; the task-specific input is placed LAST so that the
    # identical registry/constraint prefix can be reused by the KV cache (speed only, no content change).
    fixed = {k: v for k, v in state.items() if k != 'input'}
    return ("Tool registry and declared constraints (JSON):\n" + json.dumps(fixed, ensure_ascii=False) +
            "\n\nTask input:\n" + state['input'] +
            "\n\nReturn {\"tools\": [...]} listing the ids of all tools to execute.")

def schema(ids):
    return {"type": "object", "properties": {"tools": {"type": "array", "items": {"type": "string", "enum": list(ids)},
            "uniqueItems": True}}, "required": ["tools"]}

def violations(sample: Sample, chosen: set[str]) -> list[str]:
    out = []
    for a, b in [tuple(p) for p in sample.excludes]:
        if a in chosen and b in chosen: out.append(f"{a} and {b} are mutually exclusive")
    for k, need in sample.requires_any.items():
        if k in chosen and not (set(need) & chosen): out.append(f"{k} requires one of {need}")
    for k, need in sample.requires.items():
        miss = set(need) - chosen
        if k in chosen and miss: out.append(f"{k} requires {sorted(miss)}")
    if sample.budget is not None:
        c = sum(sample.costs.get(t, 0) for t in chosen)
        if c > sample.budget + 1e-9: out.append(f"total cost {c:g} exceeds budget {sample.budget:g}")
    if not out and topological_plan(sample, frozenset(chosen)) is None:
        out.append("no legal execution order exists for this set")
    return out

class Planner:
    def __init__(self, model_path, cache_dir: Path):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=8192, n_threads=2, n_batch=512, seed=2027, verbose=False)
        self.cache_dir = cache_dir; cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = Path(model_path).name
        self.calls = 0; self.prompt_tokens = 0; self.completion_tokens = 0; self.seconds = 0.0

    def ask(self, messages, ids):
        key = hashlib.sha256(json.dumps([self.model_name, messages, list(ids)], ensure_ascii=False).encode()).hexdigest()
        f = self.cache_dir / (key + '.json')
        if f.exists():
            rec = json.loads(f.read_text())
        else:
            t = time.monotonic()
            r = self.llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=160,
                    response_format={"type": "json_object", "schema": schema(ids)})
            rec = {'messages': messages, 'response': r, 'seconds': time.monotonic() - t}
            f.write_text(json.dumps(rec, ensure_ascii=False))
        self.calls += 1
        u = rec['response'].get('usage', {}); self.prompt_tokens += u.get('prompt_tokens', 0)
        self.completion_tokens += u.get('completion_tokens', 0); self.seconds += rec.get('seconds', 0)
        text = rec['response']['choices'][0]['message']['content']
        try:
            tools = json.loads(text)['tools']; parsed = True
        except Exception:
            tools = []; parsed = False
        return [t for t in tools if t in ids], parsed

    def plan(self, sample: Sample, history=None):
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_prompt(sample.public_state())}]
        if history: msgs += history
        chosen, parsed = self.ask(msgs, list(sample.options))
        return set(chosen), msgs, parsed

def recovery_sample(task, first):
    s = task.sample
    remaining = s.budget - first['cost']
    opts = {k: v for k, v in s.options.items() if k != 'read_primary'}
    anyof = {k: [x for x in v if x in opts] for k, v in s.requires_any.items() if k in opts}
    return replace(s, options=opts, gold=frozenset(), budget=remaining,
        costs={k: v for k, v in s.costs.items() if k in opts},
        excludes=[p for p in s.excludes if set(p) <= set(opts)], requires_any=anyof,
        precedes={k: [x for x in v if x in opts] for k, v in s.precedes.items() if k in opts},
        text=s.text + '\nRuntime observation: primary read failed before any records were loaded. Primary is unavailable. Remaining budget: ' + str(remaining))

def run_arm(task, planner, repair: bool, recovery: bool):
    """Returns result dict; calls are cached so shared prefixes are not recomputed."""
    def decide(sample):
        chosen, msgs, parsed = planner.plan(sample)
        log = [{'chosen': sorted(chosen), 'parsed': parsed}]
        if repair:
            v = violations(sample, chosen)
            if v:
                hist = [{'role': 'assistant', 'content': json.dumps({'tools': sorted(chosen)})},
                        {'role': 'user', 'content': 'The workflow validator rejected this set: ' + '; '.join(v) +
                         '. Return a corrected {"tools": [...]} that satisfies all constraints.'}]
                chosen, _, parsed = planner.plan(sample, hist)
                log.append({'repair_feedback': v, 'chosen': sorted(chosen), 'parsed': parsed})
        return chosen, log
    chosen, log = decide(task.sample)
    first = execute_workflow(task, chosen)
    res = {'success': first['success'], 'total_cost': first['cost'], 'replanned': False, 'attempts': [first], 'decisions': log}
    if recovery and first['failed_tool'] == 'read_primary' and not first['executed']:
        upd = recovery_sample(task, first)
        chosen2, log2 = decide(upd)
        retry = execute_workflow(replace(task, sample=upd), chosen2)
        total = first['cost'] + retry['cost']
        res = {'success': retry['success'] and total <= task.sample.budget + 1e-9, 'total_cost': total, 'replanned': True,
               'attempts': [first, retry], 'decisions': log + log2}
    return res

def main():
    split, model, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3]); limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    src = {'dev': 'data/v2_development/workflow_development.json',
           'sealed': 'results_private/v2_sealed/workflow_sealed_test.json'}[split]
    rows = json.loads(Path(src).read_text())[:limit]
    planner = Planner(model, outdir / 'cache')
    (outdir / split).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        t = WorkflowTask(Sample.from_dict(r['sample']), r['ledger'], r['request'], r['fx'], r['primary_fails'], r['difficulty'])
        target = outdir / split / (t.sample.id + '.json')
        if target.exists(): continue
        res = {name: run_arm(t, planner, rep, rec) for name, rep, rec in
               [('llm_static', False, False), ('llm_recovery', False, True), ('llm_repair_recovery', True, True)]}
        target.write_text(json.dumps({'id': t.sample.id, 'difficulty': t.difficulty, 'primary_fails': t.primary_fails,
                                      'methods': res}, ensure_ascii=False, default=str))
        print(split, i + 1, '/', len(rows), {k: v['success'] for k, v in res.items()},
              f'{planner.seconds:.0f}s', flush=True)

if __name__ == '__main__':
    main()
