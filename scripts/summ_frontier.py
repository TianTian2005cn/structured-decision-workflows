import json,glob,sys,os
B='outputs/paper03_caise/frontier'
for d in sorted(glob.glob(B+'/e*_*')):
    rs=[json.load(open(f)) for f in glob.glob(d+'/*.json')]
    if not rs:continue
    ms=list(rs[0]['methods'])
    print(os.path.basename(d),len(rs),{m:sum(bool(r['methods'][m]['success']) for r in rs) for m in ms},
          'legal',{m:sum(bool(r['methods'][m].get('valid')) for r in rs) for m in ms if m.endswith('static')})
