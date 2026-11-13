"""Frontier generative models in the closed domains (paper 3 revision).
E1 planner: DeepSeek Flash / Pro return a tool set (JSON object) from the identical public state; arms: static,
   + the same one-step recovery, + one validator-feedback repair + recovery.
E2 scorer swap: DeepSeek Flash next-token log-probabilities (Yes/No per tool, digit for count) feed the IDENTICAL frozen
   decoder (top-k / constraint routing / recovery) used with the commercial SDM.
Temperature 0, thinking disabled, exact-request cache. Prompts fixed before any call. No tuning on outcomes.
Usage: python scripts/run_frontier_closed.py <e1|e2> <model> <domain: ledger|ledger_nl|orders> [workers]"""
import json,sys,os,math,hashlib,time,re
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from jevset.schema import Sample
from jevset.algorithms import cardinality_topk,objective
from jevset.evaluation import topological_plan
from jevset.v2 import RoutePolicy,coverage_portfolio,select_route
from jevset.workflow_v2 import WorkflowTask,execute_workflow,execute_with_read_recovery
from jevset.workflow_order import make_orders,execute_order,execute_order_with_recovery
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'outputs/paper03_caise/frontier'
def key():
    for l in open(ROOT/'.env',encoding='utf-8-sig'):
        if l.startswith('DEEPSEEK_API_KEY='):return l.split('=',1)[1].strip()
KEY=key()
def call(payload):
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    f=OUT/'cache'/(h+'.json')
    if f.exists():return json.loads(f.read_text(encoding='utf-8'))['response']
    body=json.dumps(payload,ensure_ascii=False).encode()
    for a in range(5):
        try:
            req=Request('https://api.deepseek.com/chat/completions',data=body,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            t=time.time()
            with urlopen(req,timeout=120) as r:resp=json.load(r)
            f.write_text(json.dumps({'payload':payload,'response':resp,'seconds':time.time()-t},ensure_ascii=False),encoding='utf-8');return resp
        except (HTTPError,URLError,TimeoutError,ConnectionError) as e:
            if a==4:raise
            time.sleep(2**a)
SYS=("You are the planning component of a workflow engine. Given a task, a registry of tools and the declared constraints, "
     "select the exact SET of tools whose execution (in dependency order) produces the requested result. Respect every "
     "constraint: requires_any (a selected tool needs at least one of the listed tools), excludes (never select both), "
     "precedes_if_both_selected, and the total cost must not exceed the budget. Do not select tools that are unnecessary "
     "or forbidden by the instructions. Answer only with a JSON object {\"tools\": [...]}.")
def user(st):
    fixed={k:v for k,v in st.items() if k!='input'}
    return ("Tool registry and declared constraints (JSON):\n"+json.dumps(fixed,ensure_ascii=False)+"\n\nTask input:\n"+st['input']+
            "\n\nReturn {\"tools\": [...]} listing the ids of all tools to execute.")
def violations(s,ch):
    out=[]
    for p in s.excludes:
        if all(x in ch for x in p) and len(p)==2:out.append(f"{p[0]} and {p[1]} are mutually exclusive")
        elif len([x for x in p if x in ch])>1:out.append(f"at most one of {list(p)} may be selected")
    for k,need in s.requires_any.items():
        if k in ch and not set(need)&ch:out.append(f"{k} requires one of {need}")
    if s.budget is not None:
        c=sum(s.costs.get(t,0) for t in ch)
        if c>s.budget+1e-9:out.append(f"total cost {c:g} exceeds budget {s.budget:g}")
    if not out and topological_plan(s,frozenset(ch)) is None:out.append("no legal execution order exists for this set")
    return out
def plan(model,s,hist=None):
    msgs=[{'role':'system','content':SYS},{'role':'user','content':user(s.public_state())}]+(hist or [])
    r=call({'model':model,'messages':msgs,'temperature':0,'max_tokens':400,'response_format':{'type':'json_object'},'thinking':{'type':'disabled'}})
    try:tools=json.loads(r['choices'][0]['message']['content'])['tools'];ok=True
    except Exception:tools=[];ok=False
    return set(t for t in tools if t in s.options),ok,r.get('usage',{})
def lp_judge(model,s):
    st=s.public_state();fixed={k:v for k,v in st.items() if k!='input'}
    base=[{'role':'system','content':'You are a careful decision model for a workflow engine. Answer exactly in the requested format.'},
          {'role':'user','content':"Tool registry and declared constraints (JSON):\n"+json.dumps(fixed,ensure_ascii=False)+"\n\nTask input:\n"+st['input']+"\n\nQuestion: "}]
    def ask(q):
        m=[base[0],{'role':'user','content':base[1]['content']+q}]
        r=call({'model':model,'messages':m,'temperature':0,'max_tokens':1,'logprobs':True,'top_logprobs':20,'thinking':{'type':'disabled'}})
        out={}
        for d in r['choices'][0]['logprobs']['content'][0]['top_logprobs']:
            k=d['token'].strip();out[k]=max(out.get(k,-1e9),d['logprob'])  # variants like ' Yes' must not overwrite 'Yes'
        return out
    p={}
    for k in s.options:
        lp=ask(f'Is option {k} ({s.options[k]}) a correct / relevant member of the set of tools that must be executed for this task? Answer Yes or No.')
        a=math.exp(max(lp.get('Yes',-50),-50));b=math.exp(max(lp.get('No',-50),-50))
        p[k]=min(1-1e-4,max(1e-4,a/(a+b) if a+b>0 else .5))
    N=len(s.options)
    lp=ask(f'How many distinct options exactly should be selected in total? Answer with one integer from 0 to {N}.')
    kd={c:math.exp(max(lp.get(str(c),-50),-50)) for c in range(N+1)};z=sum(kd.values())
    kd={c:max(v/z,1e-6) for c,v in kd.items()};z=sum(kd.values())
    return p,{c:v/z for c,v in kd.items()}
def tasks(domain):
    if domain=='orders':return [(t,execute_order,execute_order_with_recovery,'load_order') for t in make_orders()]
    rows=json.loads((ROOT/'results_private/v2_sealed/workflow_sealed_test.json').read_text(encoding='utf-8'));out=[]
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_sealed_nl_variant import nl
    for r in rows:
        s=Sample.from_dict(r['sample'])
        if domain=='ledger_nl':
            old='Request: '+json.dumps(r['request']);s=replace(s,text=s.text.replace(old,'Request: '+nl(r['request'],s.id)))
        out.append((WorkflowTask(s,r['ledger'],r['request'],r['fx'],r['primary_fails'],r['difficulty']),execute_workflow,execute_with_read_recovery,'read_primary'))
    return out
def recovery_sample(s,first,primary):
    rem=s.budget-first['cost'];opts={k:v for k,v in s.options.items() if k!=primary}
    note={'read_primary':'primary read failed before any records were loaded. Primary is unavailable.',
          'load_order':'the primary order service failed before any order lines were loaded. It is unavailable.'}[primary]
    return replace(s,options=opts,gold=frozenset(),budget=rem,costs={k:v for k,v in s.costs.items() if k in opts},
        excludes=[p for p in s.excludes if set(p)<=set(opts)],requires_any={k:[x for x in v if x in opts] for k,v in s.requires_any.items() if k in opts},
        precedes={k:[x for x in v if x in opts] for k,v in s.precedes.items() if k in opts},text=s.text+'\nRuntime observation: '+note+' Remaining budget: '+str(rem))
def e1_item(model,t,ex,_rec,primary):
    def decide(s,repair):
        ch,ok,_=plan(model,s);log=[{'chosen':sorted(ch),'parsed':ok}]
        if repair:
            v=violations(s,ch)
            if v:
                hist=[{'role':'assistant','content':json.dumps({'tools':sorted(ch)})},{'role':'user','content':'The workflow validator rejected this set: '+'; '.join(v)+'. Return a corrected {"tools": [...]} that satisfies all constraints.'}]
                ch,ok,_=plan(model,s,hist);log.append({'feedback':v,'chosen':sorted(ch),'parsed':ok})
        return ch,log
    res={}
    for name,rep,rec in [('gen_static',False,False),('gen_recovery',False,True),('gen_repair_recovery',True,True)]:
        ch,log=decide(t.sample,rep);first=ex(t,ch)
        r={'success':first['success'],'total_cost':first['cost'],'valid':first['valid'],'error':first['error'],'replanned':False,'decisions':log}
        if rec and first['failed_tool']==primary and not first['executed']:
            upd=recovery_sample(t.sample,first,primary);ch2,log2=decide(upd,rep);retry=ex(replace(t,sample=upd),ch2);tot=first['cost']+retry['cost']
            r={'success':retry['success'] and tot<=t.sample.budget+1e-9,'total_cost':tot,'valid':first['valid'],'error':retry['error'],'replanned':True,'decisions':log+log2}
        res[name]=r
    return res
def e2_item(model,t,ex,rec,primary):
    pol=RoutePolicy(.65,.15,'constrained')
    def sel(s,mode):
        p,k=lp_judge(model,s)
        if mode=='topk':return cardinality_topk(p,k)
        d=replace(s,requires={},requires_any={},precedes={},excludes=[],budget=None) if mode=='route_constraints_off' else s
        pool=coverage_portfolio(d,p,k,width=24,legacy_slots=12);r,_=select_route(d,p,k,pol)
        return cardinality_topk(p,k) if r=='topk' else max(pool.sets,key=lambda ss:(objective(d,ss,p,k),tuple(sorted(ss))))
    res={}
    for mode in ['topk','route','route_constraints_off']:
        ch=sel(t.sample,mode);first=ex(t,ch)
        res[mode+'_static']={'success':first['success'],'valid':first['valid'],'error':first['error'],'total_cost':first['cost'],'selection':sorted(ch)}
        try:r=rec(t,lambda x:sel(x,mode))
        except RuntimeError as e:r={'success':first['success'],'total_cost':first['cost'],'replanned':False}
        res[mode+'_recovery']={'success':r['success'],'total_cost':r['total_cost'],'replanned':r['replanned']}
    return res
def main():
    exp,model,domain=sys.argv[1:4];W=int(sys.argv[4]) if len(sys.argv)>4 else 8
    (OUT/'cache').mkdir(parents=True,exist_ok=True);d=OUT/f'{exp}_{model}_{domain}';d.mkdir(parents=True,exist_ok=True)
    fn=e1_item if exp=='e1' else e2_item
    todo=[x for x in tasks(domain) if not (d/(x[0].sample.id+'.json')).exists()]
    def work(x):
        t=x[0];res=fn(model,*x)
        (d/(t.sample.id+'.json')).write_text(json.dumps({'id':t.sample.id,'difficulty':t.difficulty,'gold':sorted(t.sample.gold),'methods':res}),encoding='utf-8')
    with ThreadPoolExecutor(W) as pool:list(pool.map(work,todo))
    print(exp,model,domain,'done',len(list(d.glob('*.json'))))
if __name__=='__main__':main()
