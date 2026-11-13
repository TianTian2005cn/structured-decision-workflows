"""Evaluation with micro/macro F1, exact match, Jaccard and constraints."""
from __future__ import annotations
import math, time
from collections import defaultdict
from .schema import Sample
from .algorithms import Prediction, feasible, dependency_order


def safe_div(a,b): return a/b if b else 0.0


def topological_plan(sample:Sample, chosen:frozenset[str]) -> list[str] | None:
    """Dependency order, deterministic lexicographic tie break; None for cyclic graph or invalid set."""
    if not feasible(sample,chosen): return None
    return dependency_order(sample,chosen)


def summarize(pairs:list[tuple[Sample,Prediction]], duration_s:float=0.0) -> dict:
    if not pairs:raise ValueError('No sample/prediction pairs')
    tp=fp=fn=0
    exact=jacc=count_match=valid=simulated_success=0
    per_label=defaultdict(lambda: [0,0,0])
    examples=[]
    recalls=[]
    for sample,pred in pairs:
        truth=set(sample.gold); guessed=set(pred.chosen)
        t=len(truth & guessed); f=len(guessed-truth); n=len(truth-guessed)
        tp+=t;fp+=f;fn+=n
        exact+=(truth==guessed)
        jacc+= (len(truth&guessed)/len(truth|guessed) if truth|guessed else 1.)
        count_match+=(len(truth)==len(guessed))
        ok = feasible(sample,pred.chosen)
        if sample.task=='tools': ok = ok and topological_plan(sample,pred.chosen) is not None
        valid+=ok
        # This is ONLY a simulated goal-coverage proxy, NOT real tool execution success.
        simulated_success+=int(ok and truth <= guessed)
        for label in sample.options:
            # Macro-F1 uses the complete registered label vocabulary, including
            # labels absent from both gold and predictions (zero_division=0).
            per_label[label]
            if label in truth and label in guessed: per_label[label][0]+=1
            elif label in guessed: per_label[label][1]+=1
            elif label in truth: per_label[label][2]+=1
        if pred.candidate_sets:
            recalls.append(int(sample.gold in pred.candidate_sets))
        examples.append({'id':sample.id,'split':sample.split,'gold':sorted(truth),
              'predicted':sorted(guessed),'valid':ok,'calls':pred.calls,
              'candidate_count':pred.candidate_count,'predicted_cardinality':pred.predicted_cardinality})
    macro = sum(safe_div(2*v[0],2*v[0]+v[1]+v[2]) for v in per_label.values())/len(per_label)
    result={'n':len(pairs),'exact_match':exact/len(pairs),
       'micro_f1':safe_div(2*tp,2*tp+fp+fn), 'macro_f1':macro,
       'jaccard':jacc/len(pairs),'cardinality_accuracy':count_match/len(pairs),
       'valid_rate':valid/len(pairs),
       'simulated_goal_coverage_rate_NOT_real_execution':simulated_success/len(pairs),
       'total_requests':sum(pred.calls for _,pred in pairs),
       'mean_candidates':sum(pred.candidate_count for _,pred in pairs)/len(pairs),
       'candidate_recall':(sum(recalls)/len(recalls)) if recalls else None,
       'elapsed_seconds':round(duration_s,4), 'examples':examples}
    return result
