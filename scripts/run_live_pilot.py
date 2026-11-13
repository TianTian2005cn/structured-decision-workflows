"""Frozen benchmark pilot: validation tuning, three option orders, raw audit.

All decoders share the exact stage-one Noul+cardinality response for paired
decoder comparisons. The threshold baseline therefore includes a count question
in the request but ignores its answer. This is not a latency comparison against
an isolated Noul-only implementation. Uses three concurrent independent task runs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from jevset.algorithms import (Prediction, beam_candidates, cardinality_topk,
                                objective, powerset)
from jevset.backend import softmax
from jevset.evaluation import summarize
from jevset.live import AuditedJevBackend, digest, load_local_key
from jevset.schema import read_jsonl
from scripts.fit_relations import fit

ROOT=Path(__file__).resolve().parent.parent
TASKS=['exam','multilabel','tools']

def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def permute(sample, order):
    keys=list(sample.options)
    seed=int(hashlib.sha256(f'20261030:{sample.id}:{order}'.encode()).hexdigest()[:16],16)
    random.Random(seed).shuffle(keys)
    if sample.task=='exam':
        # Upstream raw choices can be gold-first. Shuffle EVERY order and assign
        # fresh display IDs so original option indices cannot reveal labels.
        mapping={k:f'option_{i+1}' for i,k in enumerate(keys)}
        return replace(sample, options={mapping[k]:sample.options[k] for k in keys},
                       gold=frozenset(mapping[k] for k in sample.gold))
    return replace(sample,options={k:sample.options[k] for k in keys})

def pred(chosen, scores, kd, candidates=(), calls=0):
    return Prediction(chosen,calls,len(candidates),scores,max(kd,key=kd.get) if kd else None,tuple(candidates))

def decode(sample,scores,kd,threshold,relation_weights=None,gamma=0):
    candidates=beam_candidates(sample,scores,kd,width=24,count_weight=.8)
    out={
        'threshold_fixed':pred(frozenset(k for k,v in scores.items() if v>=.5),scores,kd),
        'threshold':pred(frozenset(k for k,v in scores.items() if v>=threshold),scores,kd),
        'topk':pred(cardinality_topk(scores,kd),scores,kd),
        'constrained':pred(candidates[0],scores,kd,candidates),
    }
    if relation_weights is not None:
        rel=replace(sample,pair_weights=relation_weights)
        rc=beam_candidates(rel,scores,kd,width=24,count_weight=.8,pair_weight=gamma)
        out['relational']=pred(rc[0],scores,kd,rc)
    return out,candidates

def stage2(sample,backend,scores,kd,candidates,blend=.25):
    start=len(backend.records)
    dist=backend.choose(sample,candidates)
    aux=softmax({i:objective(sample,s,scores,kd,count_weight=.8) for i,s in enumerate(candidates)})
    fused={i:(1-blend)*math.log(max(1e-8,dist.get(i,0)))+blend*math.log(max(1e-8,aux[i])) for i in range(len(candidates))}
    return pred(candidates[max(fused,key=fused.get)],scores,kd,candidates,len(backend.records)-start)

def percentile(values,q):
    values=sorted(values)
    return values[round((len(values)-1)*q)]

def interval(vals):
    rng=random.Random(20261030)
    draws=[sum(rng.choices(vals,k=len(vals)))/len(vals) for _ in range(4000)]
    return [percentile(draws,.025),percentile(draws,.975)]

def score_rows(sample,prediction):
    m=summarize([(sample,prediction)])
    return {'id':sample.id,'gold':sorted(sample.gold),'predicted':sorted(prediction.chosen),
            'exact_match':float(sample.gold==prediction.chosen),
            'valid':m['valid_rate'],'candidate_recall':m['candidate_recall'],
            'scores':prediction.scores,'predicted_cardinality':prediction.predicted_cardinality,
            'candidate_count':prediction.candidate_count}

def run_task(task, protocol, out):
    rows=read_jsonl(str(ROOT/'data/benchmarks'/f'{task}.jsonl'))
    backend=AuditedJevBackend(cache_dir=out/'raw'/task,model=protocol['model'],
                             max_requests=400,max_input_tokens=2000000,retries=1)
    params_path=out/f'{task}_frozen.json'
    relations=fit(rows) if task=='multilabel' else None
    if relations is not None:
        write(out/'train_relations.json',{'source':'TRAIN ONLY','training_n':sum(r.split=='train' for r in rows),
                                         'pair_weights':relations})
    val=[r for r in rows if r.split=='validation']
    test=[r for r in rows if r.split=='test']
    # Only validation annotations enter hyperparameter selection. The file is
    # committed to disk before any test inference occurs and never re-tuned.
    if params_path.exists():
        params=json.loads(params_path.read_text(encoding='utf-8'))
    else:
        validation=[]
        for i,s in enumerate(val):
            scores,kd=backend.judge(s)
            validation.append((s,scores,kd))
            if (i+1)%8==0:print(f'{task}: validation {i+1}/{len(val)}',flush=True)
        metric='macro_f1' if task=='multilabel' else 'exact_match'
        candidates=[]
        for threshold in [.2,.35,.5,.65,.8]:
            pairs=[(s,decode(s,p,k,threshold)[0]['threshold']) for s,p,k in validation]
            candidates.append((summarize(pairs)[metric],-abs(threshold-.5),threshold))
        threshold=max(candidates)[2]
        gamma=0
        if relations is not None:
            candidates=[]
            for g in [0,.1,.3,.5]:
                pairs=[(s,decode(s,p,k,threshold,relations,g)[0]['relational']) for s,p,k in validation]
                candidates.append((summarize(pairs)[metric],-g,g))
            gamma=max(candidates)[2]
        params={'utc':datetime.now(timezone.utc).isoformat(),'threshold':threshold,'gamma':gamma,
                'tuning_metric':metric,'validation_ids':[s.id for s in val],
                'width':24,'count_weight':.8,'blend':.25,
                'note':'Frozen before test inference. Shared Noul+count first stage; fixed primary contrast adaptive vs threshold.'}
        write(params_path,params)
    all_results={}
    for order in protocol['orders']:
        pairs_by_method={}
        for i,original in enumerate(test):
            s=permute(original,order)
            p,kd=backend.judge(s)
            predictions,candidates=decode(s,p,kd,params['threshold'],relations,params['gamma'])
            predictions['adaptive']=stage2(s,backend,p,kd,candidates)
            if len(s.options)<=7:
                full=powerset(list(s.options))
                d=backend.choose(s,full)
                predictions['full_small_n']=pred(full[max(d,key=d.get)],p,kd,full)
            for method,prediction in predictions.items():
                pairs_by_method.setdefault(method,[]).append((s,prediction))
            if (i+1)%8==0:print(f'{task}: order {order}, test {i+1}/{len(test)}, HTTP {backend.calls}',flush=True)
        result={}
        for method,pairs in pairs_by_method.items():
            result[method]=summarize(pairs)
            result[method]['exact_match_ci95']=interval([float(s.gold==p.chosen) for s,p in pairs])
            result[method]['examples']=[score_rows(s,p) for s,p in pairs]
            result[method]['total_requests']=None
        all_results[str(order)]=result
        write(out/f'{task}_results.json',{'task':task,'parameters':params,'orders':all_results})
    # Average permutations WITHIN each item before the paired item bootstrap.
    # Permutations are not extra independent test observations.
    differences=[]
    for i in range(len(test)):
        differences.append(sum(all_results[str(o)]['adaptive']['examples'][i]['exact_match']-
                               all_results[str(o)]['threshold']['examples'][i]['exact_match']
                               for o in protocol['orders'])/len(protocol['orders']))
    primary={'contrast':'adaptive minus validation-tuned threshold',
             'metric':'exact_match','n_independent_items':len(test),
             'mean_difference':sum(differences)/len(differences),'ci95':interval(differences),
             'inference':'exploratory pilot; 3 task contrasts, no confirmatory significance claim'}
    final={'task':task,'parameters':params,'orders':all_results,'paired_primary':primary,
           'usage':{'new_http_attempts':backend.calls,'previous_http_attempts':backend.previous_requests,
                    'cache_hits_this_process':backend.cache_hits,'input_tokens':backend.input_tokens,
                    'output_tokens':backend.output_tokens,
                    'estimated_usd_at_0_042_per_million_input_tokens':backend.input_tokens*.042/1000000},
           'resolved_models':sorted({r['response'].get('model','unknown') for r in backend.records}),
           'scope':'public benchmark pilot; tool category selection, no execution/constraint success claims'}
    write(out/f'{task}_results.json',final)
    print(f'{task}: DONE; paired delta={primary["mean_difference"]:.3f}, CI={primary["ci95"]}',flush=True)
    return final

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs/live_pilot_20261030')
    ap.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    a=ap.parse_args()
    out=a.output_dir
    out.mkdir(parents=True,exist_ok=True)
    load_local_key(ROOT/'.env')
    data_manifest=json.loads((ROOT/'data/benchmarks/manifest.json').read_text(encoding='utf-8'))
    protocol={'model':'jev-1.14.0','orders':[0,1,2],'seed':20261030,
              'registered_utc':datetime.now(timezone.utc).isoformat(),
              'data_manifest_sha256':digest(data_manifest),
              'processed_data_sha256':data_manifest['processed'],
              'design':'exploratory pilot, shared mixed Noul+count stage one, fixed width24 blend.25 count.8; all three option orders shuffled, SATA IDs re-encoded after shuffling',
              'primary_contrast':'adaptive minus validation-tuned threshold exact-match; paired bootstrap by item',
              'request_cap_per_task':400,'input_token_cap_per_task':2000000,
              'price_source':'https://docs.typesafe.ai/models accessed 2026-10-30',
              'independent_identical_call_replication':'not performed; cache is exact-request memoization',
              'held_out_caveat':'These pilot test items become development data for future changes; reserve fresh confirmatory test.'}
    target=out/'protocol.json'
    if target.exists():
        saved=json.loads(target.read_text())
        same_data=saved.get('processed_data_sha256')==protocol['processed_data_sha256']
        if not same_data and saved['data_manifest_sha256']!=protocol['data_manifest_sha256']:
            raise ValueError('Dataset changed; use a new run directory')
        protocol=saved
    else:write(target,protocol)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda t:run_task(t,protocol,out),a.tasks))
    write(out/'summary.json',{'protocol':protocol,'tasks':{r['task']:r for r in results}})

if __name__=='__main__':main()
