import sys,json,os;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))));sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
from open_sdm import LexicalScorer
from jevset.algorithms import cardinality_topk,objective
from jevset.v2 import RoutePolicy,coverage_portfolio,select_route
from jevset.workflow_order import make_orders,execute_order,execute_order_with_recovery
sc=LexicalScorer();pol=RoutePolicy(.65,.15,'constrained')
def sel(s,mode):
    p,k=sc.judge(s)
    if mode=='topk':return cardinality_topk(p,k)
    pool=coverage_portfolio(s,p,k,width=24,legacy_slots=12);r,_=select_route(s,p,k,pol)
    return cardinality_topk(p,k) if r=='topk' else max(pool.sets,key=lambda ss:(objective(s,ss,p,k),tuple(sorted(ss))))
res={m:0 for m in ['topk_static','route_static','topk_recovery','route_recovery']}
for t in make_orders():
    for mode in ['topk','route']:
        res[mode+'_static']+=execute_order(t,sel(t.sample,mode))['success']
        res[mode+'_recovery']+=execute_order_with_recovery(t,lambda x:sel(x,mode))['success']
print('order lexical',res)
