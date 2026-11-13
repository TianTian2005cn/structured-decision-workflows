"""Statistics for the CAiSE paper-3 sealed confirmation (pre-registered tests in protocol.json)."""
import json,glob,math,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
D=ROOT/'outputs/paper03_caise/order_domain'
def binom_tail(b,n):return sum(math.comb(n,i) for i in range(b,n+1))/2**n if n else 1.0
def wilson(k,n,z=1.96):
    if n==0:return (0,0)
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return (c-h,c+h)
def main():
    rs=[json.load(open(f)) for f in sorted(glob.glob(str(D/'items/*.json')))]
    llm={}
    for f in glob.glob(str(D/'llm_sealed/*.json')):
        r=json.load(open(f));llm[r['id']]=r['methods']
    for r in rs:
        if r['id'] in llm:r['methods'].update(llm[r['id']])
    n=len(rs);ms=list(rs[0]['methods'])
    ok={m:[bool(r['methods'][m]['success']) for r in rs] for m in ms}
    out={'n_completed':n,'n_planned':96,'methods':{}}
    for m in ms:
        k=sum(ok[m]);lo,hi=wilson(k,n)
        out['methods'][m]={'success':k,'rate':k/n,'wilson95':[lo,hi],
            'mean_tool_cost':sum(r['methods'][m]['total_cost'] for r in rs)/n,
            'replans':sum(bool(r['methods'][m].get('replanned')) for r in rs)}
    by={}
    for r in rs:by.setdefault(r['difficulty'],[]).append(r)
    out['by_difficulty']={d:{m:sum(bool(x['methods'][m]['success']) for x in v) for m in ms}|{'n':len(v)} for d,v in by.items()}
    H=[('H1_primary','route_recovery','topk_static'),('H2','route_static','topk_static'),
       ('H3','route_recovery','route_static'),('H4','route_static','route_constraints_off_static')]
    tests=[]
    for name,a,b in H:
        b1=sum(x and not y for x,y in zip(ok[a],ok[b]));c1=sum(y and not x for x,y in zip(ok[a],ok[b]))
        tests.append({'hypothesis':name,'a':a,'b':b,'a_only':b1,'b_only':c1,'diff_pp':100*(sum(ok[a])-sum(ok[b]))/n,
                      'p_one_sided':binom_tail(b1,b1+c1)})
    order=sorted(range(len(tests)),key=lambda i:tests[i]['p_one_sided']);m=len(tests);prev=0
    for rank,i in enumerate(order):
        adj=min(1,max(prev,(m-rank)*tests[i]['p_one_sided']));prev=adj;tests[i]['p_holm']=adj;tests[i]['reject_at_.05']=adj<.05
    out['confirmatory_tests']=tests
    rng=random.Random(2027);idx=list(range(n));
    def inter(s):return sum(ok['route_recovery'][i]-ok['route_static'][i]-ok['topk_recovery'][i]+ok['topk_static'][i] for i in s)/len(s)*100
    bs=sorted(inter([rng.choice(idx) for _ in idx]) for _ in range(10000))
    out['interaction_pp']={'estimate':inter(idx),'ci95':[bs[249],bs[9749]]}
    if 'llm_repair_recovery' in ok:
        extra=[]
        for a,b in [('route_recovery','llm_repair_recovery'),('route_recovery','llm_recovery'),('route_static','llm_static')]:
            b1=sum(x and not y for x,y in zip(ok[a],ok[b]));c1=sum(y and not x for x,y in zip(ok[a],ok[b]))
            extra.append({'a':a,'b':b,'a_only':b1,'b_only':c1,'diff_pp':100*(sum(ok[a])-sum(ok[b]))/n,
                          'p_two_sided_exact':min(1,2*binom_tail(max(b1,c1),b1+c1))})
        out['sdm_vs_llm_secondary']=extra
    for f in ['workflow_usage.json']:
        if (D/f).exists():out[f]=json.load(open(D/f))
    (D/'analysis.json').write_text(json.dumps(out,indent=1))
    print(json.dumps(out,indent=1)[:3500])
main()
