import os; os.makedirs("fig", exist_ok=True); os.makedirs("gen", exist_ok=True)
import math,json
from figs_open_rows import rows  # label,role,env,units,ka,na,kb,nb,ci
def newcombe(k1,n1,k2,n2,z=1.96):
    def w(k,n):
        p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
    p1,p2=k1/n1,k2/n2;l1,u1=w(k1,n1);l2,u2=w(k2,n2);d=p1-p2
    return 100*d,100*(d-math.sqrt((p1-l1)**2+(u2-p2)**2)),100*(d+math.sqrt((u1-p1)**2+(p2-l2)**2))
from statistics import NormalDist
N=NormalDist()
out=[];lines=[]
for lab,role,env,units,a,n,b,m,ci in rows:
    d,lo,hi=newcombe(a,n,b,m)
    kind='N'
    if ci: lo,hi=ci;kind='P'
    se=(hi-lo)/3.92; mde=2.80*se
    z=d/se if se>0 else 0; p=2*(1-N.cdf(abs(z)))
    out.append(dict(label=lab,role=role,env=env,units=units,sdm=f'{a}/{n}',ctl=f'{b}/{m}',diff=d,lo=lo,hi=hi,kind=kind,mde=mde,p=p))
# Holm over 14 two-sided tests
order=sorted(range(len(out)),key=lambda i:out[i]['p']);prev=0
for r,i in enumerate(order):
    adj=min(1,max(prev,(len(out)-r)*out[i]['p']));prev=adj;out[i]['p_holm']=adj
for o in out:
    dag='$^\\dagger$' if o['kind']=='N' else ''
    lines.append(f"{o['role']} & {o['env']} & {o['units']} & {o['sdm']} & {o['ctl']} & {o['diff']:+.1f} & [{o['lo']:+.1f}, {o['hi']:+.1f}]{dag} & {o['mde']:.0f} & {min(1,o['p_holm']):.2f}\\\\")
open('gen/rq2_rows.tex','w').write('\n'.join(lines)+'\n')
json.dump(out,open('gen/rq2.json','w'),indent=1)
for o in out:print(o['label'],round(o['diff'],1),round(o['mde'],1),round(o['p_holm'],2))
