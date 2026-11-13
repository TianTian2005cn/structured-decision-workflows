"""400-task scale validation of frozen V2 strategies and 2x2 interactions."""
from pathlib import Path
from dataclasses import asdict,replace
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
import json,sys,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jevset.workflow_scale import make_scale_tasks,execute_scale,frozen_read_recovery
from jevset.v2 import RoutePolicy,coverage_portfolio,select_route
from jevset.algorithms import cardinality_topk,objective
from jevset.live import AuditedJevBackend,load_local_key,digest
from scripts.run_live_pilot import write

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/v3/workflow_scale'
FROZEN=['jevset/v2.py','jevset/algorithms.py','jevset/schema.py','jevset/workflow_v2.py']

def run_cohort(cohort,tasks):
    folder=OUT/cohort;folder.mkdir(parents=True,exist_ok=True)
    b=AuditedJevBackend(cache_dir=folder/'raw',model='jev-1.14.0',max_requests=900,max_input_tokens=6000000,retries=2)
    policy=RoutePolicy(.65,.15,'constrained');all_results=[]
    for index,task in enumerate(tasks):
        target=folder/(task.base.sample.id+'.json')
        if target.exists():all_results.append(json.loads(target.read_text(encoding='utf-8')));continue
        def selector(s,mode):
            p,k=b.judge(s)
            if mode=='topk':
                selected=cardinality_topk(p,k);last_selection['set']=selected;return selected
            decode_s=replace(s,requires={},requires_any={},precedes={},excludes=[],budget=None) if mode=='route_constraints_off' else s
            pool=coverage_portfolio(decode_s,p,k,width=24,legacy_slots=12)
            route,_=select_route(decode_s,p,k,policy)
            selected=cardinality_topk(p,k) if route=='topk' else max(pool.sets,key=lambda ss:(objective(decode_s,ss,p,k),tuple(sorted(ss))))
            last_selection['set']=selected
            return selected
        results={}
        for mode in ['topk','route','route_constraints_off']:
            for recovery in [False,True]:
                name=mode+('_recovery' if recovery else '_static')
                before=len(b.records);start=monotonic()
                last_selection={}
                if recovery:
                    try:r=frozen_read_recovery(task,lambda s:selector(s,mode))
                    except RuntimeError as exc:
                        if str(exc)!='V2 recovery supports initial read only':raise
                        first=execute_scale(task,last_selection['set'])
                        r={'success':False,'total_cost':first['cost'],'replanned':False,'attempts':[first],
                           'controller_error':'Frozen V2 refused replan after prior operation; failure retained'}
                else:
                    first=execute_scale(task,selector(task.base.sample,mode))
                    r={'success':first['success'],'total_cost':first['cost'],'replanned':False,'attempts':[first]}
                r.update({'logical_model_requests':len(b.records)-before,
                          'cached_or_live_wall_seconds':monotonic()-start})
                results[name]=r
        record={'id':task.base.sample.id,'cohort':cohort,'cluster':task.cluster,
                'fault':task.fault,'clarification_goal':task.clarify,'budget':task.base.sample.budget,
                'signature':task.signature(),'methods':results}
        write(target,record);all_results.append(record)
        if (index+1)%10==0:print(cohort,index+1,'/',len(tasks),'new HTTP',b.calls,flush=True)
    write(folder/'results.json',{'status':'DEVELOPMENT_SCALE_NOT_SEALED_TEST','cohort':cohort,'n':len(all_results),
        'examples':all_results,'input_tokens':b.input_tokens,'output_tokens':b.output_tokens,
        'http_attempts':b.calls+b.previous_requests,'resolved_models':sorted(b.resolved_models)})
    return all_results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohorts',nargs='+',default=['basic','medium','hard','transfer'],choices=['basic','medium','hard','transfer']);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'outputs/v2_development/workflow_split_manifest.json').read_text())
    excluded=set(g for values in manifest['groups'].values() for g in values)
    tasks=make_scale_tasks(excluded,100)
    # Validate the generator, not model performance; faults are disabled solely
    # in this software oracle check. No failing generated rows are dropped.
    for t in tasks:
        test=execute_scale(replace(t,fault=None),t.base.sample.gold)
        if not test['success']:raise AssertionError((t.base.sample.id,test))
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'EXPANDED DEVELOPMENT, NOT FINAL TEST',
        'n':400,'cohorts':{k:100 for k in ['basic','medium','hard','transfer']},
        'model':'jev-1.14.0','policy':asdict(RoutePolicy(.65,.15,'constrained')),'candidate_width':24,'legacy_slots':12,
        'max_replans':1,'primary_four':['topk_static','route_static','topk_recovery','route_recovery'],
        'factor_interpretation':'primary factor is V2 routing/constraint pipeline versus Top-K, not pure constraints; matched route_constraints_off is secondary',
        'frozen_algorithm_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FROZEN},
        'environment_sha256':hashlib.sha256((ROOT/'jevset/workflow_scale.py').read_bytes()).hexdigest(),
        'task_signatures':[t.signature() for t in tasks],
        'seal_protection':'All old reserved and visible workflow families excluded; no private test data read',
        'power':'400 is a design target, not a proven power minimum; paired and clustered uncertainty reported',
        'transfer_scope':'unseen tool IDs/schema/validation chains, shared numerical transaction semantics; not proof of general external-service transfer',
        'failure_policy':'primary and late failures pre-generated; frozen recovery handles initial primary read only; unsupported failures retained'}
    target=OUT/'protocol.json'
    if target.exists():
        saved=json.loads(target.read_text(encoding='utf-8'))
        if any(saved[k]!=protocol[k] for k in ['frozen_algorithm_sha256','environment_sha256','task_signatures']):raise ValueError('Frozen scale experiment changed')
    else:write(target,protocol)
    load_local_key(ROOT/'.env')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda cohort:run_cohort(cohort,[t for t in tasks if t.cohort==cohort]),a.cohorts))

if __name__=='__main__':main()
