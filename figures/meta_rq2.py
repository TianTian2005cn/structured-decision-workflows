import json,math
o=json.load(open('gen/rq2.json'))
def pool(rows):
    y=[r['diff'] for r in rows];v=[((r['hi']-r['lo'])/3.92)**2 for r in rows];w=[1/x for x in v]
    fe=sum(a*b for a,b in zip(w,y))/sum(w);Q=sum(wi*(yi-fe)**2 for wi,yi in zip(w,y));df=len(y)-1
    C=sum(w)-sum(x*x for x in w)/sum(w);tau2=max(0,(Q-df)/C)
    ws=[1/(x+tau2) for x in v];re=sum(a*b for a,b in zip(ws,y))/sum(ws);se=math.sqrt(1/sum(ws))
    return dict(k=len(y),re=re,lo=re-1.96*se,hi=re+1.96*se,tau2=tau2,Q=Q,I2=max(0,(Q-df)/Q) if Q>0 else 0)
res={'all':pool(o),'excl_metatool':pool([r for r in o if r['env']!='MetaTool']),
     'paired_only':pool([r for r in o if r['kind']=='P'])}
json.dump(res,open('gen/rq2_meta.json','w'),indent=1);print(json.dumps(res,indent=1))
