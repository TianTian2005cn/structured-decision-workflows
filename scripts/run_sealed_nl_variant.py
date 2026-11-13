"""Robustness variant (post-confirmation, labelled as such): the 96 sealed workflow tasks with the JSON request
replaced by a deterministic natural-language paraphrase. Everything else (registry, constraints, ledger, oracle,
frozen V2 algorithms and parameters) is identical. Tests whether the closed-workflow effect depends on a
machine-readable request. Template fixed before any call; no tuning."""
import json,sys,hashlib
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jevset.schema import Sample
from jevset.live import AuditedJevBackend,load_local_key,digest
from jevset.algorithms import cardinality_topk,objective
from jevset.v2 import RoutePolicy,coverage_portfolio,select_route
from jevset.workflow_v2 import WorkflowTask,execute_workflow,execute_with_read_recovery
from scripts.run_live_pilot import write
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'outputs/paper03_caise/sealed_nl_variant'
STAT={'total':'the total','average':'the average','largest':'the largest single','median':'the median'}
VERB=['Please compute','I need','Report','Calculate']
def nl(req,key):
    h=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    reg=' across all regions' if req['region'] is None else f" for the {req['region']} region"
    per=' over the whole ledger period' if not req['period'] else f" for the months {req['period'][0]} through {req['period'][1]}"
    st=', counting settled transactions only' if req['settled_only'] else ', counting transactions of every status'
    val='net (refund-adjusted)' if req['net'] else 'gross (before refunds)'
    return f"{VERB[h%4]} {STAT[req['statistic']]} {val} transaction value in USD{reg}{per}{st}, delivered as a {req['format'].upper()} report."
def main():
    rows=json.loads((ROOT/'results_private/v2_sealed/workflow_sealed_test.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True,exist_ok=True);load_local_key(ROOT/'.env')
    b=AuditedJevBackend(cache_dir=OUT/'raw',model='jev-1.14.0',max_requests=600,max_input_tokens=3000000,retries=2)
    policy=RoutePolicy(.65,.15,'constrained');(OUT/'items').mkdir(exist_ok=True)
    def selector(s,mode):
        p,k=b.judge(s)
        if mode=='topk':return cardinality_topk(p,k)
        pool=coverage_portfolio(s,p,k,width=24,legacy_slots=12);route,_=select_route(s,p,k,policy)
        return cardinality_topk(p,k) if route=='topk' else max(pool.sets,key=lambda ss:(objective(s,ss,p,k),tuple(sorted(ss))))
    for i,r in enumerate(rows):
        s=Sample.from_dict(r['sample']);old='Request: '+json.dumps(r['request'])
        assert old in s.text,(s.id)
        s=replace(s,text=s.text.replace(old,'Request: '+nl(r['request'],s.id)))
        t=WorkflowTask(s,r['ledger'],r['request'],r['fx'],r['primary_fails'],r['difficulty'])
        target=OUT/'items'/(s.id+'.json')
        if target.exists():continue
        res={}
        for mode in ['topk','route']:
            first=execute_workflow(t,selector(s,mode))
            res[mode+'_static']={'success':first['success'],'total_cost':first['cost']}
            rr=execute_with_read_recovery(t,lambda x:selector(x,mode))
            res[mode+'_recovery']={'success':rr['success'],'total_cost':rr['total_cost'],'replanned':rr['replanned']}
        write(target,{'id':s.id,'difficulty':t.difficulty,'request_text':nl(r['request'],s.id),'methods':res})
        if (i+1)%16==0:print(i+1,flush=True)
    write(OUT/'usage.json',{'http_attempts':b.calls+b.previous_requests,'input_tokens':b.input_tokens,'output_tokens':b.output_tokens})
if __name__=='__main__':main()
