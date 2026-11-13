"""Run-to-run variability of the sealed workflow evaluation (run 1 = confirmatory; runs 2-5 fresh caches)."""
import json,glob,statistics
from pathlib import Path
B=Path(__file__).resolve().parent.parent/'outputs/paper03_caise'
dirs=['sealed_confirmation','sealed_replication','sealed_run3','sealed_run4','sealed_run5']
runs=[{json.load(open(f))['id']:json.load(open(f)) for f in glob.glob(str(B/d/'workflow/*.json'))} for d in dirs]
ids=sorted(runs[0]);ms=[m for m in runs[0][ids[0]]['methods'] if not m.startswith('llm')]
out={'n_runs':len(runs),'n':len(ids),'methods':{}}
for m in ms:
    succ=[sum(bool(r[i]['methods'][m]['success']) for i in ids) for r in runs]
    stable=sum(len({bool(r[i]['methods'][m]['success']) for r in runs})==1 for i in ids)
    out['methods'][m]={'successes':succ,'mean':statistics.mean(succ),'sd':statistics.stdev(succ),'min':min(succ),'max':max(succ),
                       'items_identical_outcome_all_runs':stable}
sel={}
for mode in ['topk','route']:
    sel[mode]=sum(len({tuple(r[i]['selections'][mode]) for r in runs})==1 for i in ids)
out['items_identical_static_selection_all_runs']=sel
diff=[]
for k in range(len(runs)):
    a=sum(runs[k][i]['methods']['route_recovery']['success'] for i in ids);b=sum(runs[k][i]['methods']['topk_static']['success'] for i in ids)
    diff.append(a-b)
out['route_recovery_minus_topk_static_per_run']=diff
(B/'sealed_runs_variability.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))
