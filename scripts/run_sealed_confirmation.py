"""CAiSE 2027 paper 3: one-shot confirmatory evaluation on the RESERVED V2 sealed sets.

Opens only results_private/v2_sealed/workflow_sealed_test.json and tools.jsonl.
Algorithm sources must hash-match the frozen V2 development protocol; otherwise abort.
The confirmatory protocol (hypotheses, primary contrast, test, multiplicity) is written
BEFORE the first sealed request and the run refuses to start if it later changes.
Each sealed item is evaluated once; no method, threshold or prompt is changed afterwards.
"""
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jevset.schema import Sample
from jevset.live import AuditedJevBackend,load_local_key,digest
from jevset.algorithms import cardinality_topk,objective
from jevset.v2 import RoutePolicy,coverage_portfolio,select_route,choose_pool
from jevset.workflow_v2 import WorkflowTask,execute_workflow,execute_with_read_recovery
from jevset.evaluation import summarize
from jevset.algorithms import Prediction
from scripts.run_live_pilot import permute,write

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/paper03_caise/sealed_confirmation'
SEALED=ROOT/'results_private/v2_sealed'
FROZEN=['jevset/v2.py','jevset/algorithms.py','jevset/schema.py','jevset/workflow_v2.py']

PROTOCOL={
 'stage':'CONFIRMATORY_SEALED_TEST_ONE_SHOT',
 'model':'jev-1.14.0',
 'frozen_sources_must_match':'outputs/v2_live_development/protocol.json code_sha256',
 'workflow':{
   'data':'results_private/v2_sealed/workflow_sealed_test.json (96 reserved goal families, never evaluated before)',
   'methods':['topk_static','route_static','topk_recovery','route_recovery',
              'route_constraints_off_static','route_constraints_off_recovery'],
   'policy':{'min_count_probability':.65,'min_boundary_margin':.15,'uncertain_action':'constrained','width':24,'legacy_slots':12,'max_replans':1},
   'primary_metric':'numeric goal success (independent oracle)',
   'hypotheses':{
     'H1_primary':'route_recovery > topk_static',
     'H2':'route_static > topk_static',
     'H3':'route_recovery > route_static',
     'H4':'route_static > route_constraints_off_static (matched decoder-constraint effect)'},
   'test':'exact one-sided McNemar (binomial on discordant pairs), Holm correction over H1-H4 at alpha=.05',
   'secondary':'interaction s11-s10-s01+s00, bootstrap 95% CI (10k, seed 2027); tool cost units; Jev requests/tokens',
   'development_reference':'32 V2 dev families: 71.9/84.4/78.1/100.0 percent; 400 V3 scale rows: 39.5/72.5/41.2/84.0 percent'},
 'metatool':{
   'data':'results_private/v2_sealed/tools.jsonl (96 reserved MetaTool items, 47-tool catalog)',
   'methods':['threshold','topk','v2_cover','v2_route'],
   'frozen_params':'outputs/v2_live_development/tools_frozen.json (threshold .8, legacy_slots 16, policy 0/0)',
   'order':'permute(sample,0), identical to development',
   'prediction':'no public hard constraints => routing collapses to Top-K (equivalence, descriptive); Exact Match reported with Wilson CI'},
 'no_peeking':'Gold labels are read only by the evaluator after selections are stored per item. Results are final regardless of outcome.'}

def frozen_check():
    saved=json.loads((ROOT/'outputs/v2_live_development/protocol.json').read_text(encoding='utf-8'))['code_sha256']
    for f in FROZEN:
        if digest((ROOT/f).read_text(encoding='utf-8'))!=saved[f]:raise SystemExit('FROZEN SOURCE CHANGED: '+f)

def run_workflow(limit=None):
    rows=json.loads((SEALED/'workflow_sealed_test.json').read_text(encoding='utf-8'))[:limit]
    tasks=[WorkflowTask(Sample.from_dict(r['sample']),r['ledger'],r['request'],r['fx'],r['primary_fails'],r['difficulty']) for r in rows]
    b=AuditedJevBackend(cache_dir=OUT/'raw_workflow',model='jev-1.14.0',max_requests=600,max_input_tokens=3000000,retries=2)
    policy=RoutePolicy(.65,.15,'constrained');folder=OUT/'workflow';folder.mkdir(parents=True,exist_ok=True)
    def selector(s,mode):
        p,k=b.judge(s)
        if mode=='topk':return cardinality_topk(p,k)
        d=replace(s,requires={},requires_any={},precedes={},excludes=[],budget=None) if mode=='route_constraints_off' else s
        pool=coverage_portfolio(d,p,k,width=24,legacy_slots=12)
        route,_=select_route(d,p,k,policy)
        return cardinality_topk(p,k) if route=='topk' else max(pool.sets,key=lambda ss:(objective(d,ss,p,k),tuple(sorted(ss))))
    for i,t in enumerate(tasks):
        target=folder/(t.sample.id+'.json')
        if target.exists():continue
        res={};sel={}
        for mode in ['topk','route','route_constraints_off']:
            before=len(b.records);chosen=selector(t.sample,mode);sel[mode]=sorted(chosen)
            first=execute_workflow(t,chosen)
            res[mode+'_static']={'success':first['success'],'total_cost':first['cost'],'replanned':False,'attempts':[first],
                                 'logical_model_requests':len(b.records)-before}
            before=len(b.records)
            try:r=execute_with_read_recovery(t,lambda s:selector(s,mode))
            except RuntimeError as exc:
                r={'success':first['success'],'total_cost':first['cost'],'replanned':False,'attempts':[first],'controller_error':str(exc)}
            r['logical_model_requests']=len(b.records)-before
            res[mode+'_recovery']=r
        write(target,{'id':t.sample.id,'family':digest(t.request),'difficulty':t.difficulty,'budget':t.sample.budget,
                      'primary_fails':t.primary_fails,'selections':sel,'methods':res})
        if (i+1)%8==0:print('workflow',i+1,'/',len(tasks),'new HTTP',b.calls,flush=True)
    write(OUT/'workflow_usage.json',{'http_attempts':b.calls+b.previous_requests,'input_tokens':b.input_tokens,
          'output_tokens':b.output_tokens,'resolved_models':sorted(b.resolved_models)})

def run_metatool(limit=None):
    rows=[json.loads(l) for l in open(SEALED/'tools.jsonl',encoding='utf-8')][:limit]
    params=json.loads((ROOT/'outputs/v2_live_development/tools_frozen.json').read_text(encoding='utf-8'))
    thr=params['threshold_from_old_development'];slots=params['legacy_slots'];policy=RoutePolicy(**params['policy'])
    b=AuditedJevBackend(cache_dir=OUT/'raw_metatool',model='jev-1.14.0',max_requests=400,max_input_tokens=3000000,retries=2)
    folder=OUT/'metatool';folder.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(rows):
        target=folder/(r['id']+'.json')
        if target.exists():continue
        s=permute(Sample.from_dict(r),0);before=len(b.records)
        p,k=b.judge(s)
        pool=coverage_portfolio(s,p,k,width=24,threshold=thr,legacy_slots=slots)
        route,feat=select_route(s,p,k,policy)
        sel={'threshold':sorted(l for l,v in p.items() if v>=thr),'topk':sorted(cardinality_topk(p,k)),
             'v2_cover':sorted(choose_pool(s,b,p,k,pool)),
             'v2_route':sorted(cardinality_topk(p,k) if route=='topk' else max(pool.sets,key=lambda ss:(objective(s,ss,p,k),tuple(sorted(ss)))))}
        write(target,{'id':r['id'],'route':route,'selections':sel,'gold':sorted(r['gold']),'requests':len(b.records)-before})
        if (i+1)%8==0:print('metatool',i+1,'/',len(rows),'new HTTP',b.calls,flush=True)
    write(OUT/'metatool_usage.json',{'http_attempts':b.calls+b.previous_requests,'input_tokens':b.input_tokens,
          'output_tokens':b.output_tokens,'resolved_models':sorted(b.resolved_models)})

def main():
    frozen_check();OUT.mkdir(parents=True,exist_ok=True)
    proto=dict(PROTOCOL,frozen_sha=[digest((ROOT/f).read_text(encoding='utf-8')) for f in FROZEN],
               sealed_sha256={f:hashlib.sha256((SEALED/f).read_bytes()).hexdigest() for f in ['workflow_sealed_test.json','tools.jsonl']},
               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=OUT/'protocol.json'
    if target.exists():
        saved=json.loads(target.read_text(encoding='utf-8'))
        if any(saved[k]!=proto[k] for k in proto if k!='created_utc'):raise SystemExit('Confirmatory protocol changed; refusing')
    else:
        proto['created_utc']=datetime.now(timezone.utc).isoformat();write(target,proto);print('protocol frozen');
        if '--freeze-only' in sys.argv:return
    load_local_key(ROOT/'.env')
    if 'workflow' in sys.argv:run_workflow()
    if 'metatool' in sys.argv:run_metatool()

if __name__=='__main__':main()
