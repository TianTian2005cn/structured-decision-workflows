"""New development calls and full ablations; sealed V2 test is never opened.

Defaults evaluate 32 NEW development items per task with one shuffled order.
This is an algorithm-development experiment, not the final independent test.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jevset.schema import read_jsonl,Sample
from jevset.live import AuditedJevBackend,load_local_key,digest
from jevset.algorithms import (Prediction,beam_candidates,cardinality_topk,objective)
from jevset.evaluation import summarize
from jevset.v2 import (coverage_portfolio,stratified_candidates,hierarchical_candidates,
                       grouped_judge,RoutePolicy,select_route,choose_pool)
from jevset.baselines_v2 import direct_choice,choice_threshold
from jevset.workflow_v2 import WorkflowTask,execute_workflow,execute_with_read_recovery
from scripts.run_live_pilot import permute,write

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/v2_live_development'
DEV=ROOT/'data/v2_development'

def prediction(chosen,p,kd,pool=None):
    return Prediction(chosen,0,len(pool.sets) if pool else 0,p,max(kd,key=kd.get) if kd else None,
                      tuple(pool.sets) if pool else ())

def backend(task):return AuditedJevBackend(cache_dir=OUT/'raw'/task,model='jev-1.14.0',max_requests=500,max_input_tokens=4000000,retries=1)
def usage(b):return {'http_attempts':b.previous_requests+b.calls,'input_tokens':b.input_tokens,'output_tokens':b.output_tokens,
                     'estimated_usd_at_previous_verified_price':b.input_tokens*.042/1000000}

def public_task(task,limit):
    rows=read_jsonl(str(DEV/f'{task}.jsonl'))
    calibration=[permute(s,0) for s in rows if s.split=='validation']
    development=[permute(s,0) for s in rows if s.split=='development'][:limit]
    d=json.loads((ROOT/f'outputs/v2_development/{task}_diagnosis.json').read_text(encoding='utf-8'))
    threshold=json.loads((ROOT/f'outputs/live_pilot_20261030/{task}_frozen.json').read_text(encoding='utf-8'))['threshold']
    b=backend(task);params_path=OUT/f'{task}_frozen.json'
    policy=RoutePolicy(**{k:d['frozen_router'][k] for k in ['min_count_probability','min_boundary_margin','uncertain_action']})
    if params_path.exists():params=json.loads(params_path.read_text(encoding='utf-8'))
    else:
        params={'threshold_from_old_development':threshold,'legacy_slots':d['frozen_legacy_slots'],
                'policy':{'min_count_probability':policy.min_count_probability,'min_boundary_margin':policy.min_boundary_margin,
                          'uncertain_action':policy.uncertain_action},'frozen_utc':datetime.now(timezone.utc).isoformat()}
        if task=='exam':
            pairs=[]
            for i,s in enumerate(calibration):
                p,none=direct_choice(b,s);pairs.append((s,p))
                if (i+1)%8==0:print('exam: direct Choice calibration',i+1,flush=True)
            scores=[]
            for t in [.005,.01,.025,.05,.1,.2,.35,.5]:
                acc=sum(choice_threshold(p,t)==s.gold for s,p in pairs)
                scores.append((acc,-t,t))
            params['direct_threshold']=max(scores)[2]
            params['direct_grid']=[{'threshold':t,'correct':acc,'n':len(pairs)} for acc,_,t in scores]
        write(params_path,params)
    priors=json.loads((ROOT/'outputs/v2_development/group_priors.json').read_text(encoding='utf-8')) if task=='multilabel' else None
    methods={};traces=[]
    for i,s in enumerate(development):
        p,k=b.judge(s)
        old=beam_candidates(s,p,k,width=24)
        from jevset.v2 import CandidatePool
        oldpool=CandidatePool(old,['v1']*len(old))
        pool=coverage_portfolio(s,p,k,width=24,threshold=threshold,legacy_slots=params['legacy_slots'])
        pure=stratified_candidates(s,p,k,width=24,threshold=threshold,sparse=task=='multilabel')
        selections={
            'threshold':prediction(frozenset(l for l,v in p.items() if v>=threshold),p,k),
            'topk':prediction(cardinality_topk(p,k),p,k),
            'v1_adaptive':prediction(choose_pool(s,b,p,k,oldpool),p,k,oldpool),
            'v2_cover':prediction(choose_pool(s,b,p,k,pool),p,k,pool),
            'v2_stratified_map':prediction(max(pure.sets,key=lambda ss:(objective(s,ss,p,k),tuple(sorted(ss)))),p,k,pure)}
        route,features=select_route(s,p,k,policy)
        routed=cardinality_topk(p,k) if route=='topk' else max(pool.sets,key=lambda ss:(objective(s,ss,p,k),tuple(sorted(ss))))
        selections['v2_route']=prediction(routed,p,k,pool)
        trace={'id':s.id,'route':route,'features':features}
        if task=='exam':
            direct,none=direct_choice(b,s)
            selections['direct_choice_threshold']=prediction(choice_threshold(direct,params['direct_threshold']),direct,{})
            trace['direct_none_probability']=none
        if priors:
            hp,hscore=hierarchical_candidates(s,p,priors,threshold=threshold)
            selections['hierarchy_train_prior_same_scores']=prediction(max(hp.sets,key=lambda ss:(hscore(ss),tuple(sorted(ss)))),p,{},hp)
            gp,gates,counts=grouped_judge(b,s)
            selections['grouped_fine_threshold_control']=prediction(frozenset(l for l,v in gp.items() if v>=threshold),gp,{})
            for name,gcount,ggate in [('grouped_train_prior_control',None,None),
                                      ('grouped_local_count_no_gate',counts,None),
                                      ('hierarchy_v2',counts,gates)]:
                hp,hscore=hierarchical_candidates(s,gp,priors,threshold=threshold,group_counts=gcount,group_scores=ggate)
                selections[name]=prediction(max(hp.sets,key=lambda ss:(hscore(ss),tuple(sorted(ss)))),gp,{},hp)
        for method,pr in selections.items():methods.setdefault(method,[]).append((s,pr))
        traces.append(trace)
        # Predictions checkpointed after every item. Resume uses raw cache.
        write(OUT/f'{task}_checkpoint.json',{'completed_ids':[t['id'] for t in traces],
              'metrics':{m:summarize(ps) for m,ps in methods.items()}})
        if (i+1)%8==0:print(task,'new development',i+1,'/',len(development),'HTTP',b.calls,flush=True)
    result={'status':'NEW_DEVELOPMENT_ONLY_NOT_SEALED_TEST','n_independent_items':len(development),
            'orders':1,'parameters':params,'methods':{m:summarize(ps) for m,ps in methods.items()},
            'trace':traces,'usage':usage(b),'resolved_models':sorted({r['response'].get('model') for r in b.records})}
    write(OUT/f'{task}_results.json',result)
    print(task,'DONE',flush=True)
    return result

def workflow_task(limit):
    # This path is deliberately fixed to visible development; no test switch.
    rows=json.loads((DEV/'workflow_development.json').read_text(encoding='utf-8'))[:limit]
    tasks=[WorkflowTask(Sample.from_dict(r['sample']),r['ledger'],r['request'],r['fx'],r['primary_fails'],r['difficulty']) for r in rows]
    b=backend('workflow');policy=RoutePolicy(.65,.15,'constrained')
    parameters={'policy':'prespecified heuristic, not fitted on MetaTool test results',
                'min_count_probability':.65,'min_boundary_margin':.15,'uncertain_action':'constrained','max_replans':1}
    write(OUT/'workflow_frozen.json',parameters)
    records=[]
    def select(s,mode):
        p,k=b.judge(s)
        if mode=='topk':return cardinality_topk(p,k)
        pool=coverage_portfolio(s,p,k,width=24,legacy_slots=12)
        route,_=select_route(s,p,k,policy)
        if route=='topk':return cardinality_topk(p,k)
        return max(pool.sets,key=lambda ss:(objective(s,ss,p,k),tuple(sorted(ss))))
    for i,t in enumerate(tasks):
        top=select(t.sample,'topk');routed=select(t.sample,'route')
        results={'topk_static':execute_workflow(t,top),'route_static':execute_workflow(t,routed),
                 'topk_recovery':execute_with_read_recovery(t,lambda s:select(s,'topk')),
                 'route_recovery':execute_with_read_recovery(t,lambda s:select(s,'route'))}
        records.append({'id':t.sample.id,'family':digest(t.request),'difficulty':t.difficulty,
                        'budget':t.sample.budget,'topk_selection':sorted(top),'route_selection':sorted(routed),'results':results})
        write(OUT/'workflow_checkpoint.json',records)
        if (i+1)%8==0:print('workflow',i+1,'/',len(tasks),'HTTP',b.calls,flush=True)
    metrics={}
    for name in records[0]['results']:
        rs=[r['results'][name] for r in records]
        metrics[name]={'n':len(rs),'success_rate':sum(bool(r['success']) for r in rs)/len(rs),
                       'mean_cost':sum(r.get('total_cost',r.get('cost',0)) for r in rs)/len(rs),
                       'replans':sum(r.get('replanned',False) for r in rs)}
    result={'status':'SYNTHETIC_WORKFLOW_DEVELOPMENT_NOT_HELDOUT','parameters':parameters,'methods':metrics,
            'examples':records,'usage':usage(b),'resolved_models':sorted({r['response'].get('model') for r in b.records})}
    write(OUT/'workflow_results.json',result);print('workflow DONE',metrics,flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=32)
    ap.add_argument('--tasks',nargs='+',choices=['exam','multilabel','tools','workflow'],default=['exam','multilabel','tools','workflow'])
    a=ap.parse_args()
    if not 1<=a.limit<=64:ap.error('limit must be 1..64')
    OUT.mkdir(parents=True,exist_ok=True);load_local_key(ROOT/'.env')
    source_files=['jevset/v2.py','jevset/algorithms.py','jevset/schema.py','jevset/workflow_v2.py','jevset/baselines_v2.py',
                  'scripts/run_v2_development.py']
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'DEVELOPMENT_ONLY',
        'model':'jev-1.14.0','limit':a.limit,'order':[0],
        'inputs':{t:digest(json.loads((ROOT/'outputs/v2_development/sealed_test_manifest.json').read_text())['tasks'][t]['processed_sha256']) for t in ['exam','multilabel','tools']},
        'code_sha256':{f:digest((ROOT/f).read_text(encoding='utf-8')) for f in source_files},
        'frozen_prior':digest(json.loads((ROOT/'outputs/v2_development/group_priors.json').read_text(encoding='utf-8'))),
        'budget_per_task':{'max_http_attempts':500,'max_input_tokens':4000000},
        'choice_funnel':'native generative baseline implemented; no local model supplied; NOT evaluated with Jev',
        'sealed_test_access':'No private files are opened by this runner.'}
    target=OUT/'protocol.json'
    if target.exists():
        saved=json.loads(target.read_text(encoding='utf-8'))
        if any(saved[k]!=protocol[k] for k in ['limit','inputs','code_sha256','frozen_prior']):
            raise ValueError('Frozen development run inputs/code changed; use an explicitly versioned run directory')
    else:write(target,protocol)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda t:workflow_task(a.limit) if t=='workflow' else public_task(t,a.limit),a.tasks))

if __name__=='__main__':main()
