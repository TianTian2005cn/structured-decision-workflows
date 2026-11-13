"""Fit symmetric label interaction weights from TRAIN data only. Small-data illustrative estimator."""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevset.schema import read_jsonl

def fit(records, smoothing:float=1., limit:float=2.):
    records=[r for r in records if r.split=='train']
    if not records:raise ValueError('No train rows (refusing validation/test leakage)')
    labels=sorted(set().union(*(set(r.options) for r in records)))
    cnt=Counter(k for r in records for k in r.gold)
    both=Counter('|'.join((a,b)) for r in records for i,a in enumerate(sorted(r.gold)) for b in sorted(r.gold)[i+1:])
    n=len(records); result={}
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            # Smoothed pointwise mutual information: P(ab)/(P(a) P(b)).
            p_ab=(both[f'{a}|{b}']+smoothing)/(n+2*smoothing)
            p_a=(cnt[a]+smoothing)/(n+2*smoothing)
            p_b=(cnt[b]+smoothing)/(n+2*smoothing)
            result[f'{a}|{b}']=round(max(-limit,min(limit,math.log(p_ab/(p_a*p_b)))),6)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    r=fit(read_jsonl(str(args.data)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'source':'TRAIN ONLY','pair_weights':r},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Saved {len(r)} train-only pair relationships to {args.output}')
if __name__=='__main__':main()
