"""V2 research decoders. Candidate builders never inspect evaluation labels.

These are development hypotheses, not validated improvements. All defaults must
be frozen on development data before any new held-out experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math

from .algorithms import (Prediction, feasible, cardinality_topk, log_bernoulli,
                         objective, beam_candidates)
from .backend import clamp, softmax


def keyset(s):return tuple(sorted(s))


def kbest_by_count(scores, per_count=24):
    """Exact top-M Bernoulli subsets in EACH cardinality stratum (no constraints).

    Dynamic programming retains M states per (processed prefix, count), so a
    low-scoring sparse prefix cannot be eliminated by dense prefixes. Ties use
    sorted label IDs; no label-truth/cardinality oracle is consulted.
    """
    if not 1<=per_count<=255:raise ValueError('per_count must be 1..255')
    dp={0:[(0.,frozenset())]}
    for label in sorted(scores):
        p=clamp(scores[label]);new={}
        for k,states in dp.items():
            for value,chosen in states:
                new.setdefault(k,[]).append((value+math.log(1-p),chosen))
                new.setdefault(k+1,[]).append((value+math.log(p),chosen|{label}))
        dp={k:sorted(v,key=lambda item:(-item[0],keyset(item[1])))[:per_count] for k,v in new.items()}
    return {k:[s for _,s in states] for k,states in dp.items()}


def dependency_closure(sample,chosen,scores):
    """Conservative prerequisite closure; no guarantee of optimal repair."""
    result=set(chosen)
    while True:
        prior=set(result)
        for label in sorted(prior):
            result.update(sample.requires.get(label,[]))
            alternatives=sample.requires_any.get(label,[])
            if alternatives and not set(alternatives)&result:
                result.add(max(alternatives,key=lambda k:(scores.get(k,0),-sample.costs.get(k,1),k)))
        if prior==result:break
        if not result<=set(sample.options):return None
    result=frozenset(result)
    return result if feasible(sample,result) else None


@dataclass
class CandidatePool:
    sets: list[frozenset]
    provenance: list[str]


def stratified_candidates(sample,scores,kdist,*,width=24,threshold=.5,sparse=False,extra_seeds=()):
    """Cardinality-stratified pool with threshold/Top-K/empty-set anchors.

    Equal-width ablation against V1 is required; the union of old and new pools
    would increase cost and is not called an equal-budget improvement.
    """
    if not 4<=width<=255:raise ValueError('V2 width must be 4..255')
    if set(scores)!=set(sample.options):raise ValueError('Scores must cover the complete option registry')
    by_k=kbest_by_count(scores,per_count=width)
    chosen=[];provenance=[];seen=set()
    def add(s,source):
        if s is not None and s not in seen and len(chosen)<width and feasible(sample,s):
            seen.add(s);chosen.append(s);provenance.append(source)
    add(dependency_closure(sample,cardinality_topk(scores,kdist),scores),'topk_anchor')
    for t in dict.fromkeys([threshold,.5]):
        add(dependency_closure(sample,frozenset(k for k,p in scores.items() if p>=t),scores),f'threshold_{t}')
    add(frozenset(),'empty_anchor')
    for s,source in extra_seeds:add(dependency_closure(sample,s,scores),source)
    n=len(scores)
    # The count prior influences allocation BEFORE pruning, not just re-ranking.
    qsum=sum(kdist.values())
    q={k:(kdist.get(k,0)/qsum if qsum else 1/(n+1)) for k in range(n+1)}
    # Broad support protects against a confidently wrong count prediction.
    weights={k:.6*q[k]+.4/(n+1) for k in range(n+1)}
    if sparse:
        norm=sum(1/(k+1)**2 for k in range(n+1))
        weights={k:.5*q[k]+.5/(k+1)**2/norm for k in range(n+1)}
    allocated={k:0 for k in by_k}
    per_stratum_cap=max(2,math.ceil(width/4))
    # Mandatory sparse strata are a prespecified hypothesis, not true |gold|.
    first=[0,1,2,3] if sparse else []
    for k in first:
        if k in by_k:
            for s in by_k[k]:
                if feasible(sample,s):add(s,f'count_{k}');break
    while len(chosen)<width:
        eligible=[k for k,v in by_k.items() if allocated[k]<min(len(v),per_stratum_cap)]
        if not eligible:break
        # Diminishing per-stratum quota prevents a sharp q from taking all slots.
        k=max(eligible,key=lambda k:(weights[k]/(allocated[k]+1),-k))
        s=by_k[k][allocated[k]];allocated[k]+=1
        add(s,f'count_{k}')
    return CandidatePool(chosen,provenance)


def coverage_portfolio(sample,scores,kdist,*,width=24,threshold=.5,legacy_slots=12):
    """Same total candidate budget; reserve a validation-chosen legacy quota."""
    if not 0<=legacy_slots<=width:raise ValueError('Legacy quota outside candidate budget')
    old=beam_candidates(sample,scores,kdist,width=width)
    new=stratified_candidates(sample,scores,kdist,width=width,threshold=threshold)
    final=[];sources=[]
    for s,source in ([(s,'legacy_prefix') for s in old[:legacy_slots]]+
                     list(zip(new.sets,new.provenance))+[(s,'legacy_fill') for s in old]):
        if s not in final and len(final)<width:final.append(s);sources.append(source)
    return CandidatePool(final,sources)


# Official GoEmotions Ekman grouping is a semantic decomposition, not a hard
# label ontology; neutral is not declared mutually exclusive with emotions.
EMOTION_GROUPS={
    'anger':['anger','annoyance','disapproval'], 'disgust':['disgust'],
    'fear':['fear','nervousness'],
    'joy':['joy','amusement','approval','excitement','gratitude','love','optimism',
           'relief','pride','admiration','desire','caring'],
    'sadness':['sadness','disappointment','embarrassment','grief','remorse'],
    'surprise':['surprise','realization','confusion','curiosity'], 'neutral':['neutral']}


def fit_group_priors(records,groups=EMOTION_GROUPS):
    train=[r for r in records if r.split=='train']
    if not train:raise ValueError('Group priors require training rows only')
    frequency=Counter(k for r in train for k in r.gold)
    priors={}
    for name,labels in groups.items():
        count=Counter(len(set(labels)&r.gold) for r in train)
        denom=len(train)+len(labels)+1
        priors[name]={k:(count[k]+1)/denom for k in range(len(labels)+1)}
    return {'source':'TRAIN ONLY','n':len(train),'frequencies':dict(frequency),
            'group_cardinality':priors}


def hierarchical_candidates(sample,scores,priors,*,width=24,threshold=.8,
                            group_counts=None,group_scores=None,rare_slots=3):
    """Soft group-count composition plus reserved rare-label singleton anchors.

    No coarse group is hard-gated out. Every fine label participates in local
    search, including labels in a group with low coarse confidence. Group count
    distributions may be train priors or new grouped Jev responses; report which.
    """
    if priors.get('source')!='TRAIN ONLY':raise ValueError('Unaudited group priors')
    if not 8<=width<=255:raise ValueError('Hierarchy width must be 8..255')
    groups={g:[l for l in labels if l in scores] for g,labels in EMOTION_GROUPS.items()}
    assigned=set().union(*(set(v) for v in groups.values()))
    if assigned!=set(scores):raise ValueError('Emotion grouping must cover all labels exactly')
    group_counts=group_counts or priors['group_cardinality']
    def q(g,k):
        dist=group_counts[g]
        return float(dist.get(k,dist.get(str(k),0)))
    def group_term(g,chosen):
        count=len(set(groups[g])&chosen)
        value=.5*math.log(max(1e-8,q(g,count)))
        if group_scores is not None:
            p=clamp(group_scores[g]);value+=.25*math.log(p if count else 1-p)
        return value
    def score(chosen):
        return log_bernoulli(scores,chosen)+sum(group_term(g,chosen) for g in groups)
    composed=[frozenset()]
    for g,labels in groups.items():
        if not labels:continue
        local_scores={l:scores[l] for l in labels}
        byk=kbest_by_count(local_scores,per_count=4)
        local=[s for k in sorted(byk) for s in byk[k]]
        # Preserve local empty + every local cardinality; global combination is
        # still approximate and candidate recall must be measured separately.
        expansions={a|b for a in composed for b in local}
        processed=set().union(*(set(groups[h]) for h in list(groups)[:list(groups).index(g)+1]))
        active={h for h in groups if set(groups[h])<=processed}
        def partial(s):
            return log_bernoulli({k:scores[k] for k in sorted(processed)},s)+sum(group_term(h,s) for h in sorted(active))
        composed=sorted(expansions,key=lambda s:(-partial(s),keyset(s)))[:width]
    # Rare protection reserves capacity, never inserts a true held-out label.
    freq=priors['frequencies']
    rare=sorted(scores,key=lambda l:(-(scores[l]/math.sqrt(freq.get(l,0)+1)),l))[:rare_slots]
    extra=[(frozenset({l}),f'rare_anchor_{l}') for l in rare]
    pool=stratified_candidates(sample,scores,{},width=width,threshold=threshold,sparse=True,extra_seeds=extra)
    seeds=list(zip(pool.sets[:min(8,len(pool.sets))],pool.provenance[:8]))
    final=[];sources=[]
    for s,source in seeds+[(s,'group_composition') for s in sorted(composed,key=lambda s:(-score(s),keyset(s)))]+list(zip(pool.sets,pool.provenance)):
        if s not in final and feasible(sample,s) and len(final)<width:final.append(s);sources.append(source)
    return CandidatePool(final,sources),score


def grouped_judge(backend,sample):
    """One typed request: fine Noul, soft group gates and within-group counts."""
    if not hasattr(backend,'_request'):raise ValueError('Grouped live inference requires a Jev-compatible backend')
    keys=list(sample.options);groups=EMOTION_GROUPS
    if set().union(*(set(v) for v in groups.values()))!=set(keys):raise ValueError('Expected 28-label GoEmotions registry')
    qs={f'fine_{i}':{'type':'noul','instructions':f'Does the author express {k}? {sample.options[k]}'} for i,k in enumerate(keys)}
    for i,(g,labels) in enumerate(groups.items()):
        qs[f'group_{i}']={'type':'noul','instructions':f'Does the author express any of these labels: {", ".join(labels)}?'}
        qs[f'count_{i}']={'type':'choice','instructions':f'How many distinct labels from this group apply: {", ".join(labels)}? Count only this group.',
                         'criteria':{f'k{k}':f'Exactly {k} labels from this group' for k in range(len(labels)+1)}}
    ans=backend._request(sample,qs)
    scores={k:float(ans[f'fine_{i}']['noul']) for i,k in enumerate(keys)}
    gates={g:float(ans[f'group_{i}']['noul']) for i,g in enumerate(groups)}
    if any(not math.isfinite(v) or not 0<=v<=1 for v in [*scores.values(),*gates.values()]):raise ValueError('Invalid grouped Noul scores')
    counts={}
    for i,(g,labels) in enumerate(groups.items()):
        raw=backend._probabilities(ans[f'count_{i}'],[f'k{k}' for k in range(len(labels)+1)])
        counts[g]={k:raw[f'k{k}'] for k in range(len(labels)+1)}
    return scores,gates,counts


@dataclass(frozen=True)
class RoutePolicy:
    min_count_probability: float=.65
    min_boundary_margin: float=.15
    uncertain_action: str='constrained'


def route_features(sample,scores,kdist):
    top=cardinality_topk(scores,kdist);ranked=sorted(scores.values(),reverse=True);k=len(top)
    margin=(ranked[k-1]-ranked[k]) if 0<k<len(ranked) else (1-ranked[0] if k==0 else ranked[-1])
    z=sum(kdist.values());q={j:p/z for j,p in kdist.items()} if z else {}
    entropy=-sum(p*math.log(p) for p in q.values() if p>0)/max(1,math.log(len(scores)+1))
    return {'count_probability':max(q.values(),default=0.),'boundary_margin':margin,
            'count_entropy':entropy,'topk_valid':feasible(sample,top),'predicted_k':k}


def select_route(sample,scores,kdist,policy=RoutePolicy()):
    if policy.uncertain_action not in {'constrained','choice'}:raise ValueError('Unknown uncertain action')
    f=route_features(sample,scores,kdist)
    if f['topk_valid'] and f['count_probability']>=policy.min_count_probability and f['boundary_margin']>=policy.min_boundary_margin:
        return 'topk',f
    return policy.uncertain_action,f


def choose_pool(sample,backend,scores,kdist,pool,blend=.25):
    d=backend.choose(sample,pool.sets)
    aux=softmax({i:objective(sample,s,scores,kdist) for i,s in enumerate(pool.sets)})
    fused={i:(1-blend)*math.log(max(1e-8,d.get(i,0)))+blend*math.log(max(1e-8,aux[i])) for i in aux}
    return pool.sets[max(fused,key=fused.get)]


def predict_v2(sample,backend,*,method='cover_v2',threshold=.5,width=24,
               priors=None,policy=RoutePolicy(),legacy_slots=12):
    first=backend.calls
    if method=='hierarchy_v2':
        scores,gates,counts=grouped_judge(backend,sample)
        pool,score=hierarchical_candidates(sample,scores,priors,width=width,threshold=threshold,
                                           group_counts=counts,group_scores=gates)
        chosen=max(pool.sets,key=lambda s:(score(s),tuple(reversed(keyset(s)))))
        kd={};route='hierarchical_map'
    else:
        scores,kd=backend.judge(sample)
        pool=coverage_portfolio(sample,scores,kd,width=width,threshold=threshold,legacy_slots=legacy_slots)
        if method=='cover_v2':chosen=choose_pool(sample,backend,scores,kd,pool);route='choice'
        elif method=='route_v2':
            route,_=select_route(sample,scores,kd,policy)
            if route=='topk':chosen=cardinality_topk(scores,kd)
            elif route=='constrained':chosen=max(pool.sets,key=lambda s:(objective(sample,s,scores,kd),keyset(s)))
            else:chosen=choose_pool(sample,backend,scores,kd,pool)
        else:raise ValueError('Unknown V2 method')
    prediction=Prediction(chosen,backend.calls-first,len(pool.sets),scores,
                          max(kd,key=kd.get) if kd else None,tuple(pool.sets))
    return prediction,{'route':route,'candidate_provenance':pool.provenance}
