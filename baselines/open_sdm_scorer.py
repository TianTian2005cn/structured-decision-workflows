"""Scorer-swap experiment (reviewer W2 / M2 / 3.1): an OPEN-WEIGHTS structured decision component built from
next-token probabilities of Qwen2.5-7B-Instruct (Q4_K_M, llama.cpp), plugged into the IDENTICAL frozen
decoder (cardinality top-k, 24-candidate coverage portfolio, constraint-aware MAP objective, routing policy
.65/.15, one-step recovery) that is used with the commercial SDM. Only the scorer changes.

Primitives mirror the commercial SDM questions:
  relevance(option) = P(Yes) / (P(Yes)+P(No)) for "Should option X be selected? Answer Yes or No."
  count distribution over 0..|options| from the next-token digit distribution (two-step for 10..16).
The model sees exactly Sample.public_state() (registry first, task input last, for KV-prefix reuse).
Also provides a LEXICAL (no-ML) scorer with the same decoder.  Greedy/argmax-free: probabilities only.
Usage: python open_sdm.py <split dev|sealed|sealed_nl> <scorer qwen|lexical> <outdir> [limit]
"""
from __future__ import annotations
import json, sys, math, hashlib, time, re
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevset.schema import Sample
from jevset.algorithms import cardinality_topk, objective
from jevset.v2 import RoutePolicy, coverage_portfolio, select_route
from jevset.workflow_v2 import WorkflowTask, execute_workflow, execute_with_read_recovery

import os
MODEL = os.environ.get('QWEN_GGUF', 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf')
SYS = 'You are a careful decision model for a workflow engine. Answer exactly in the requested format.'

class QwenScorer:
    name = 'qwen2.5-7b-logit'
    def __init__(self, cache: Path):
        from llama_cpp import Llama
        self.llm = Llama(model_path=MODEL, n_ctx=6144, n_threads=2, n_batch=512, seed=2027, verbose=False, logits_all=False)
        self.cache = cache; cache.mkdir(parents=True, exist_ok=True)
        tok = lambda s: self.llm.tokenize(s.encode(), add_bos=False, special=True)
        self.tok = tok
        self.YES, self.NO = tok('Yes')[0], tok('No')[0]
        self.DIG = [tok(str(d))[0] for d in range(10)]
        for t in [self.YES, self.NO] + self.DIG: assert isinstance(t, int)
        self.seconds = 0.

    def _next_logits(self, n_prefix, suffix_tokens):
        self.llm.n_tokens = n_prefix
        self.llm.eval(suffix_tokens)
        import numpy as np
        return np.array(np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm.n_vocab(),)), dtype=np.float64)

    def judge(self, s: Sample):
        st = s.public_state()
        key = hashlib.sha256(json.dumps(['qwen-v1', st], sort_keys=True).encode()).hexdigest()
        f = self.cache / (key + '.json')
        if f.exists():
            r = json.loads(f.read_text()); return r['p'], {int(k): v for k, v in r['k'].items()}
        t0 = time.monotonic()
        fixed = {k: v for k, v in st.items() if k != 'input'}
        head = ('<|im_start|>system\n' + SYS + '<|im_end|>\n<|im_start|>user\nTool registry and declared constraints (JSON):\n'
                + json.dumps(fixed, ensure_ascii=False) + '\n\nTask input:\n' + st['input'] + '\n\nQuestion: ')
        tail = '<|im_end|>\n<|im_start|>assistant\n'
        ptoks = self.tok(head)
        self.llm.reset(); self.llm.eval(ptoks); n = self.llm.n_tokens
        p = {}
        for k in s.options:
            q = f'Is option {k} ({s.options[k]}) a correct / relevant member of the set of tools that must be executed for this task? Answer Yes or No.'
            lg = self._next_logits(n, self.tok(q + tail))
            a, b = lg[self.YES], lg[self.NO]; m = max(a, b)
            p[k] = float(math.exp(a - m) / (math.exp(a - m) + math.exp(b - m)))
        N = len(s.options)
        q = f'How many distinct options exactly should be selected in total? Answer with one integer from 0 to {N}.'
        sfx = self.tok(q + tail)
        lg = self._next_logits(n, sfx)
        dl = [float(lg[t]) for t in self.DIG]; m = max(dl); e = [math.exp(x - m) for x in dl]; z = sum(e); first = [x / z for x in e]
        kd = {d: first[d] for d in range(min(9, N) + 1)}
        if N >= 10:
            lg2 = self._next_logits(n, sfx + [self.DIG[1]])
            full = lg2 - lg2.max(); ex = [math.exp(float(v)) for v in full]; Z = sum(ex)
            pdig = [ex[t] / Z for t in self.DIG]
            stop = max(0., 1 - sum(pdig))
            sec = [pdig[j] for j in range(0, min(9, N - 10) + 1)]
            zz = stop + sum(sec)
            kd[1] = first[1] * stop / zz
            for j, v in enumerate(sec): kd[10 + j] = first[1] * v / zz
        for c in range(N + 1): kd.setdefault(c, 0.)
        tot = sum(kd.values()); kd = {c: v / tot for c, v in kd.items()}
        self.seconds += time.monotonic() - t0
        f.write_text(json.dumps({'p': p, 'k': kd, 'seconds': time.monotonic() - t0, 'prefix_tokens': n}))
        return p, kd

class LexicalScorer:
    """No-ML baseline: token overlap between the task input and each option's id+meaning; count = #options
    whose overlap exceeds the median (peaked distribution). Uses the same decoder."""
    name = 'lexical'
    def judge(self, s: Sample):
        words = lambda t: set(w for w in re.findall(r'[a-z]+', t.lower()) if len(w) > 2)
        inp = words(s.text.split('Ledger:')[0])
        ov = {k: len(words(k.replace('_', ' ') + ' ' + v) & inp) / max(1, len(words(k.replace('_', ' ') + ' ' + v))) for k, v in s.options.items()}
        p = {k: min(.99, max(.01, v)) for k, v in ov.items()}
        thr = sorted(ov.values())[len(ov) // 2]
        c = sum(v > thr for v in ov.values())
        kd = {j: math.exp(-0.5 * (j - c) ** 2) for j in range(len(ov) + 1)}; z = sum(kd.values())
        return p, {j: v / z for j, v in kd.items()}

def load(split):
    src = {'dev': 'data/v2_development/workflow_development.json',
           'sealed': 'results_private/v2_sealed/workflow_sealed_test.json',
           'sealed_nl': 'results_private/v2_sealed/workflow_sealed_test.json'}[split]
    rows = json.loads(Path(src).read_text())
    out = []
    for r in rows:
        s = Sample.from_dict(r['sample'])
        if split == 'sealed_nl':
            from nl_variant import nl
            old = 'Request: ' + json.dumps(r['request']); assert old in s.text
            s = replace(s, text=s.text.replace(old, 'Request: ' + nl(r['request'], s.id)))
        out.append(WorkflowTask(s, r['ledger'], r['request'], r['fx'], r['primary_fails'], r['difficulty']))
    return out

def main():
    split, which, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3]); limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    scorer = QwenScorer(outdir / 'cache') if which == 'qwen' else LexicalScorer()
    policy = RoutePolicy(.65, .15, 'constrained'); (outdir / split).mkdir(parents=True, exist_ok=True)
    def selector(s, mode):
        p, k = scorer.judge(s)
        if mode == 'topk': return cardinality_topk(p, k)
        d = replace(s, requires={}, requires_any={}, precedes={}, excludes=[], budget=None) if mode == 'route_constraints_off' else s
        pool = coverage_portfolio(d, p, k, width=24, legacy_slots=12); route, _ = select_route(d, p, k, policy)
        return cardinality_topk(p, k) if route == 'topk' else max(pool.sets, key=lambda ss: (objective(d, ss, p, k), tuple(sorted(ss))))
    for i, t in enumerate(load(split)[:limit]):
        target = outdir / split / (t.sample.id + '.json')
        if target.exists(): continue
        res = {}
        for mode in ['topk', 'route', 'route_constraints_off']:
            ch = selector(t.sample, mode); first = execute_workflow(t, ch)
            res[mode + '_static'] = {'success': first['success'], 'total_cost': first['cost'], 'valid': first['valid'], 'error': first['error'], 'selection': sorted(ch)}
            try: r = execute_with_read_recovery(t, lambda x: selector(x, mode))
            except RuntimeError as exc: r = {'success': first['success'], 'total_cost': first['cost'], 'replanned': False, 'controller_error': str(exc)}
            res[mode + '_recovery'] = {'success': r['success'], 'total_cost': r['total_cost'], 'replanned': r['replanned']}
        target.write_text(json.dumps({'id': t.sample.id, 'difficulty': t.difficulty, 'scorer': scorer.name, 'methods': res}))
        print(split, which, i + 1, {k: v['success'] for k, v in res.items()}, f"{getattr(scorer, 'seconds', 0):.0f}s", flush=True)

if __name__ == '__main__':
    main()
