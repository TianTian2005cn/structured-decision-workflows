import json,glob,math
from pathlib import Path
D=Path(__file__).resolve().parent.parent/'outputs/paper03_caise/sealed_confirmation'
rs=[json.load(open(f)) for f in sorted(glob.glob(str(D/'metatool/*.json')))]
def wilson(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
out={'n':len(rs),'routes':{r:sum(x['route']==r for x in rs) for r in set(x['route'] for x in rs)},'methods':{}}
for m in rs[0]['selections']:
    em=sum(set(x['selections'][m])==set(x['gold']) for x in rs)
    tp=sum(len(set(x['selections'][m])&set(x['gold'])) for x in rs);fp=sum(len(set(x['selections'][m])-set(x['gold'])) for x in rs);fn=sum(len(set(x['gold'])-set(x['selections'][m])) for x in rs)
    out['methods'][m]={'exact':em,'em_rate':em/len(rs),'wilson95':wilson(em,len(rs)),'micro_f1':2*tp/(2*tp+fp+fn),
        'card_acc':sum(len(x['selections'][m])==len(x['gold']) for x in rs)/len(rs)}
out['route_equals_topk']=sum(x['selections']['v2_route']==x['selections']['topk'] for x in rs)
out['usage']=json.load(open(D/'metatool_usage.json'))
(D/'metatool_analysis.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
