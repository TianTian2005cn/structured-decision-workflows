"""Second closed domain (order fulfilment and invoicing), pre-registered one-shot evaluation of the frozen SDM
decision pipeline. Same frozen decoder/policy/recovery as the ledger domain; nothing tuned on this domain.
Protocol (hypotheses H1-H4, exact one-sided McNemar, Holm) is frozen before the first request."""
import json,sys,hashlib
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jevset.live import AuditedJevBackend,load_local_key,digest
from jevset.algorithms import cardinality_topk,objective
from jevset.v2 import RoutePolicy,coverage_portfolio,select_route
from jevset.workflow_order import make_orders,execute_order,execute_order_with_recovery
from scripts.run_live_pilot import write
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'outputs/paper03_caise/order_domain'
FROZEN=['jevset/v2.py','jevset/algorithms.py','jevset/schema.py']
def main():
    saved=json.loads((ROOT/'outputs/v2_live_development/protocol.json').read_text(encoding='utf-8'))['code_sha256']
    for f in FROZEN:
        if digest((ROOT/f).read_text(encoding='utf-8'))!=saved[f]:raise SystemExit('FROZEN SOURCE CHANGED '+f)
    tasks=make_orders();OUT.mkdir(parents=True,exist_ok=True)
    proto={'stage':'PRE_REGISTERED_SECOND_DOMAIN_ONE_SHOT','model':'jev-1.14.0','n':len(tasks),
      'methods':['topk_static','route_static','topk_recovery','route_recovery','route_constraints_off_static','route_constraints_off_recovery'],
      'hypotheses':{'H1_primary':'route_recovery > topk_static','H2':'route_static > topk_static','H3':'route_recovery > route_static',
                    'H4':'route_static > route_constraints_off_static'},
      'test':'exact one-sided McNemar, Holm over H1-H4, alpha .05','policy':'.65/.15 constrained, width 24, legacy 12, one replan',
      'environment_sha256':hashlib.sha256((ROOT/'jevset/workflow_order.py').read_bytes()).hexdigest(),
      'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'task_ids':[t.sample.id for t in tasks],
      'design_note':'Domain, rules and constraints written before any SDM call on this domain; gold sets verified only by the oracle self-test.'}
    target=OUT/'protocol.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if any(old[k]!=proto[k] for k in proto):raise SystemExit('protocol changed; refusing')
    else:
        proto['created_utc']=datetime.now(timezone.utc).isoformat();write(target,proto)
        if '--freeze-only' in sys.argv:print('frozen');return
    load_local_key(ROOT/'.env')
    b=AuditedJevBackend(cache_dir=OUT/'raw',model='jev-1.14.0',max_requests=600,max_input_tokens=3000000,retries=2)
    policy=RoutePolicy(.65,.15,'constrained');(OUT/'items').mkdir(exist_ok=True)
    def selector(s,mode):
        p,k=b.judge(s)
        if mode=='topk':return cardinality_topk(p,k)
        d=replace(s,requires={},requires_any={},precedes={},excludes=[],budget=None) if mode=='route_constraints_off' else s
        pool=coverage_portfolio(d,p,k,width=24,legacy_slots=12);route,_=select_route(d,p,k,policy)
        return cardinality_topk(p,k) if route=='topk' else max(pool.sets,key=lambda ss:(objective(d,ss,p,k),tuple(sorted(ss))))
    for i,t in enumerate(tasks):
        f=OUT/'items'/(t.sample.id+'.json')
        if f.exists():continue
        res={};sel={}
        for mode in ['topk','route','route_constraints_off']:
            ch=selector(t.sample,mode);sel[mode]=sorted(ch);first=execute_order(t,ch)
            res[mode+'_static']={'success':first['success'],'total_cost':first['cost'],'valid':first['valid'],'error':first['error']}
            r=execute_order_with_recovery(t,lambda x:selector(x,mode))
            res[mode+'_recovery']={'success':r['success'],'total_cost':r['total_cost'],'replanned':r['replanned']}
        write(f,{'id':t.sample.id,'difficulty':t.difficulty,'gold':sorted(t.sample.gold),'selections':sel,'methods':res})
        if (i+1)%16==0:print(i+1,flush=True)
    write(OUT/'usage.json',{'http_attempts':b.calls+b.previous_requests,'input_tokens':b.input_tokens,'output_tokens':b.output_tokens,'models':sorted(b.resolved_models)})
if __name__=='__main__':main()
