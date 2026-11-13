"""Sensitivity of the offloadable cost share rho to the strictness of the candidate constructor (paper 3, CAiSE).
Levels: strict = next action actually built as a complete candidate by the frozen constructor (pre-call info only);
binder  = every parameter field of the logged tool call(s) is bounded (enum/bool/small int/default) or copyable
          from earlier public messages, i.e. K1 holds in principle for a better binder;
tool    = any call that emits native tool calls (an omniscient constructor for tool actions; text never closed).
Shares are cost-weighted (re-priced usage) and call-weighted; 95% CIs by bootstrap over trajectories (10k)."""
import json,random,collections
from pathlib import Path
R=Path(__file__).resolve().parent.parent
S=json.load(open(R/'outputs/direct_v3/actor_states.json'))
def lv(s):
    pf=s.get('parameter_fields') or {}
    tool=s['role'] in ('single_tool','multi_tool_batch')
    binder=tool and pf.get('free',0)==0 and s.get('unknown_parameter_fields',0)==0
    return {'strict':bool(s['exact_single_action_covered']),'binder':bool(binder or s['exact_single_action_covered']),'tool':tool}
envs=sorted(set(s['benchmark'] for s in S))+['all']
out={}
rng=random.Random(2027)
for env in envs:
    rows=[s for s in S if env=='all' or s['benchmark']==env]
    traj=collections.defaultdict(list)
    for s in rows:traj[(s['cohort'],s['method'],s['case'])].append(s)
    keys=list(traj)
    def share(ks,level,w):
        num=den=0.
        for k in ks:
            for s in traj[k]:
                x=s['cost_high'] if w=='cost' else 1.
                den+=x;num+=x*lv(s)[level]
        return num/den
    res={'calls':len(rows),'trajectories':len(keys)}
    for level in ['strict','binder','tool']:
        for w in ['cost','calls']:
            est=share(keys,level,w)
            bs=sorted(share([rng.choice(keys) for _ in keys],level,w) for _ in range(2000))
            res[f'{level}_{w}']=[round(100*est,2),round(100*bs[50],2),round(100*bs[1949],2)]
    out[env]=res
(R/'outputs/paper03_caise/rho_sensitivity.json').write_text(json.dumps(out,indent=1))
for e,r in out.items():print(e,r)
