"""Algorithms A-F. No gold labels read here."""
from __future__ import annotations
import itertools, math
from dataclasses import dataclass
from .backend import DecisionBackend, clamp, softmax
from .schema import Sample


def dependency_order(sample: Sample,chosen: frozenset[str]) -> list[str] | None:
    """Topological execution order with conjunctive and alternative providers.

    OR dependencies require one completed provider, not every selected provider;
    this admits an acyclic alternative even when another alternative cycles.
    Optional precedence is active only when both operations are selected.
    """
    deps={k:set(sample.requires.get(k,[])) for k in chosen}
    for before,afters in sample.precedes.items():
        if before in chosen:
            for after in afters:
                if after in chosen:deps[after].add(before)
    done=set();ordered=[]
    while deps:
        ready=sorted(k for k,values in deps.items() if values<=done and
            (k not in sample.requires_any or bool(set(sample.requires_any[k])&done)))
        if not ready:return None
        k=ready[0];ordered.append(k);done.add(k);del deps[k]
    return ordered


def feasible(sample: Sample, chosen: frozenset[str]) -> bool:
    if not chosen <= set(sample.options):
        return False
    if any(k in chosen and not set(required) <= chosen for k,required in sample.requires.items()):
        return False
    if any(k in chosen and not set(alternatives)&chosen for k,alternatives in sample.requires_any.items()):
        return False
    if any(set(pair) <= chosen for pair in sample.excludes):
        return False
    if sample.budget is not None and sum(sample.costs.get(k,1) for k in chosen) > sample.budget+1e-9:
        return False
    return dependency_order(sample,chosen) is not None


def powerset(labels: list[str], max_options: int=7) -> list[frozenset[str]]:
    if len(labels) > max_options or 2**len(labels) > 255:
        raise ValueError(f'Full powerset unavailable: n={len(labels)}, limit is min({max_options}, 7)')
    return [frozenset(s) for r in range(len(labels)+1) for s in itertools.combinations(labels,r)]


def cardinality_topk(scores: dict[str,float], k_dist: dict[int,float]) -> frozenset[str]:
    k = max(k_dist, key=lambda k:(k_dist[k],-k)) if k_dist else round(sum(scores.values()))
    k = max(0, min(k, len(scores)))
    return frozenset(sorted(scores,key=lambda l:(-scores[l],l))[:k])


def log_bernoulli(scores: dict[str,float], chosen: frozenset[str]) -> float:
    return sum(math.log(clamp(p if k in chosen else 1-p)) for k,p in scores.items())


def pair_score(sample: Sample, chosen: frozenset[str]) -> float:
    return sum(w for ab,w in sample.pair_weights.items() if set(ab.split('|')) <= chosen)


def objective(sample: Sample, chosen: frozenset[str], scores: dict[str,float], kdist: dict[int,float],
              *, count_weight:float=0.8, pair_weight:float=0) -> float:
    if not feasible(sample, chosen):
        return float('-inf')
    cardinality = math.log(max(1e-8,kdist.get(len(chosen), 1/max(1,len(scores)+1)))) if kdist else 0
    return log_bernoulli(scores,chosen) + count_weight*cardinality + pair_weight*pair_score(sample,chosen)


def beam_candidates(sample: Sample, scores:dict[str,float], kdist:dict[int,float],
                    *, width:int=24, count_weight:float=.8, pair_weight:float=0) -> list[frozenset[str]]:
    """Beam on partial subsets; filter feasibility only at the END to allow requires edges."""
    if width<1 or width>255:
        raise ValueError('beam width must be 1..255')
    labels = sorted(scores,key=lambda x:(abs(scores[x]-.5),x),reverse=True)
    beam = [frozenset()]
    for label in labels:
        expansions = []
        for state in beam:
            expansions.extend([state, state|{label}])
        # Partial Bernoulli objective; global cardinality added only after full depth.
        seen=set()
        ranked=[]
        processed = set(labels[:labels.index(label)+1])
        for state in expansions:
            if state in seen: continue
            seen.add(state)
            partial=math.fsum(math.log(clamp(scores[k] if k in state else 1-scores[k])) for k in sorted(processed))
            # Penalize violations that cannot be repaired (excluded or overspent).
            if any(set(pair)<=state for pair in sample.excludes): continue
            if sample.budget is not None and sum(sample.costs.get(k,1) for k in state)>sample.budget: continue
            ranked.append((partial,state))
        beam = [s for _,s in sorted(ranked,key=lambda p:(-p[0],tuple(sorted(p[1]))))[:width]]
        if not beam:
            beam=[frozenset()]
    scored = [(objective(sample,s,scores,kdist,count_weight=count_weight,pair_weight=pair_weight),s) for s in beam]
    valid = [(val,s) for val,s in scored if math.isfinite(val)]
    # If beam pruning removed all feasible leaves (requires constraints), fallback to bounded exact for small n.
    if not valid and len(labels)<=12:
        allsets=[frozenset(sub) for r in range(len(labels)+1) for sub in itertools.combinations(labels,r)]
        valid=[(objective(sample,s,scores,kdist,count_weight=count_weight,pair_weight=pair_weight),s)
               for s in allsets if feasible(sample,s)]
    if not valid:
        valid=[(objective(sample,frozenset(),scores,kdist),frozenset())]
    valid.sort(key=lambda p:(-p[0],tuple(sorted(p[1]))))
    return [s for _,s in valid[:min(width,255)]]


def constrained_map(sample:Sample,scores:dict[str,float], kdist:dict[int,float],
                    *, width:int=40, count_weight:float=.8,pair_weight:float=0) -> frozenset[str]:
    return beam_candidates(sample,scores,kdist,width=width,count_weight=count_weight,pair_weight=pair_weight)[0]


@dataclass
class Prediction:
    chosen: frozenset[str]
    calls: int
    candidate_count: int
    scores: dict[str,float]
    predicted_cardinality: int | None
    candidate_sets: tuple[frozenset[str], ...] = ()


def predict(sample:Sample, backend:DecisionBackend, *, method:str='adaptive',
            threshold:float=.5, width:int=24, blend:float=.25, count_weight:float=.8,
            pair_weight:float=0, max_full:int=7) -> Prediction:
    """Methods: threshold(A), topk(C), full(B), adaptive(D), constrained(E), chain(F).
    A/B do not enforce constraints, intentionally exposing violations as a baseline.
    """
    first=backend.calls
    scores:dict[str,float] = {}
    kdist:dict[int,float] = {}
    n_candidates=0
    candidate_sets=[]
    if method=='full':
        candidate_sets=powerset(list(sample.options),max_full)
        d=backend.choose(sample,candidate_sets)
        chosen=candidate_sets[max(d,key=d.get)]
        n_candidates=len(candidate_sets)
    elif method=='chain':
        selected=set()
        scores={}
        for key in sample.options:
            p=backend.conditional_noul(sample.with_selected(selected),key)
            scores[key]=p
            if p>=threshold:
                selected.add(key)
        chosen=frozenset(selected)
    else:
        scores,kdist=backend.judge(sample, cardinality=(method != 'threshold'))
        if method=='threshold':
            chosen=frozenset(k for k,p in scores.items() if p>=threshold)
        elif method=='topk':
            chosen=cardinality_topk(scores,kdist)
        elif method=='constrained':
            chosen=constrained_map(sample,scores,kdist,width=width,count_weight=count_weight,pair_weight=pair_weight)
        elif method=='adaptive':
            candidate_sets=beam_candidates(sample,scores,kdist,width=width,count_weight=count_weight,pair_weight=pair_weight)
            n_candidates=len(candidate_sets)
            if len(candidate_sets)==1:
                chosen=candidate_sets[0]
            else:
                choice_dist=backend.choose(sample,candidate_sets)
                aux=softmax({i:objective(sample,s,scores,kdist,count_weight=count_weight,pair_weight=pair_weight)
                             for i,s in enumerate(candidate_sets)})
                # Product-of-experts between stage-2 Choice and stage-1 prior.
                fused={i:(1-blend)*math.log(max(1e-8,choice_dist.get(i,0))) +
                         blend*math.log(max(1e-8,aux[i])) for i in range(len(candidate_sets))}
                chosen=candidate_sets[max(fused,key=fused.get)]
        else:
            raise ValueError(f'Unknown method: {method}')
    pred_k=max(kdist,key=kdist.get) if kdist else None
    return Prediction(chosen=chosen,calls=backend.calls-first,candidate_count=n_candidates,
                      scores=scores,predicted_cardinality=pred_k,candidate_sets=tuple(candidate_sets))
