"""Harder synthetic tool execution: optional filters, units, refunds, recovery.

The oracle checks report content from original records independently of tool
gold. No email/network/calendar tools execute. Recovery is limited to a failed
initial read and reselects at most once under the remaining budget.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import csv,io,json,random,statistics
from .schema import Sample
from .evaluation import topological_plan

REGISTRY={
 'read_primary':'Read the supplied transaction ledger from the primary local source. It may fail at runtime.',
 'read_backup':'Read an identical offline backup ledger; costs two units, useful if primary is unavailable.',
 'filter_region':'Keep only the requested region, if the request specifies a region.',
 'filter_period':'Keep only transactions in the requested month range, if specified.',
 'filter_settled':'Keep only settled transactions; exclude pending and cancelled transactions.',
 'subtract_refunds':'Convert gross revenue to net revenue by subtracting each transaction refund in its original currency.',
 'convert_usd':'Convert each current transaction value to USD using supplied offline FX rates.',
 'sum_values':'Calculate total of current transaction values.',
 'mean_values':'Calculate arithmetic mean of current transaction values.',
 'max_values':'Calculate the largest individual current transaction value.',
 'median_values':'Calculate median current transaction value.',
 'report_json':'Write the computed result and requested metadata as an in-memory JSON report.',
 'report_csv':'Write the computed result and requested metadata as an in-memory CSV report.',
 'round_rows':'Round every individual transaction to the nearest dollar before aggregation. Can introduce rounding error.',
 'plot_preview':'Create a chart preview. Does not produce the requested machine-readable numeric report.',
 'send_external_email':'Send a report externally. Unavailable in this local benchmark; no external action is permitted.'}
AGGREGATORS={'total':'sum_values','average':'mean_values','largest':'max_values','median':'median_values'}
READERS=['read_primary','read_backup']
FILTERS=['filter_region','filter_period','filter_settled']
TRANSFORMS=['subtract_refunds','convert_usd','round_rows']
REPORTS=['report_json','report_csv']

@dataclass
class WorkflowTask:
    sample:Sample
    ledger:list[dict]
    request:dict
    fx:dict
    primary_fails:bool=False
    difficulty:str='standard'


def make_workflows(n=48,seed=20261031,split='development'):
    rng=random.Random(seed);out=[]
    for i in range(n):
        # Goal families are deliberately varied, including empty selections,
        # near-budget limits, distractors and an unannounced read failure.
        region=[None,'East','West','North'][i%4]
        period=None if (i//4)%2==0 else ['2026-07','2026-08']
        req={'region':region,'period':period,'settled_only':bool((i//8)%2),
             'net':bool((i//16)%2),'statistic':list(AGGREGATORS)[(i//32)%4],
             'format':['json','csv'][(i//128)%2],'currency':'USD'}
        ledger=[]
        for j in range(24):
            r=['East','West','North'][j%3]
            ledger.append({'region':r,'month':['2026-06','2026-07','2026-08','2026-09'][j%4],
                'status':['settled','settled','pending','cancelled'][j%4],
                'currency':{'East':'EUR','West':'USD','North':'JPY'}[r],
                'gross':rng.randint(10000,90000)/100,'refund':rng.randint(0,3000)/100})
        fx={'USD':1.,'EUR':1.137,'JPY':.00673}
        needed={'read_primary',AGGREGATORS[req['statistic']],'report_'+req['format']}
        if req['region'] is not None:needed.add('filter_region')
        if req['period'] is not None:needed.add('filter_period')
        if req['settled_only']:needed.add('filter_settled')
        if req['net']:needed.add('subtract_refunds')
        if req['region']!='West':needed.add('convert_usd')
        anyof={k:READERS for k in FILTERS+TRANSFORMS+list(AGGREGATORS.values())}
        anyof.update({k:list(AGGREGATORS.values()) for k in REPORTS+['plot_preview','send_external_email']})
        precedence={k:TRANSFORMS+list(AGGREGATORS.values()) for k in FILTERS}
        precedence['subtract_refunds']=['convert_usd','round_rows']+list(AGGREGATORS.values())
        precedence['convert_usd']=['round_rows']+list(AGGREGATORS.values())
        precedence['round_rows']=list(AGGREGATORS.values())
        excludes=[READERS,REPORTS]
        aggs=list(AGGREGATORS.values())
        excludes += [[a,b] for x,a in enumerate(aggs) for b in aggs[x+1:]]
        costs={k:1. for k in REGISTRY};costs['read_backup']=2.;costs['send_external_email']=20.
        fail=(i%6==5)
        # One unit failed primary + two-unit backup requires two extra units.
        budget=len(needed)+(2 if fail else (i%3))
        text=('Create exactly the requested machine-readable numeric report. Do not round individual rows or send external messages. '
              'Empty input has value 0 for total and null for other statistics. '
              'Use backup only when necessary. All tools operate at most once.\nRequest: '+json.dumps(req)+
              '\nOffline FX rates to USD: '+json.dumps(fx)+'\nLedger: '+json.dumps(ledger))
        sample=Sample(f'workflow-{seed}-{i:03d}','tools',text,REGISTRY.copy(),frozenset(needed),split=split,
                       requires_any=anyof,precedes=precedence,excludes=excludes,costs=costs,budget=budget)
        out.append(WorkflowTask(sample,ledger,req,fx,fail,'read_failure' if fail else 'tight_budget' if i%3==0 else 'distractors'))
    return out


def expected_report(task):
    """Independent reference implementation, no Sample.gold or selected tools."""
    r=task.request
    values=[]
    for row in task.ledger:
        if r['region'] is not None and row['region']!=r['region']:continue
        if r['period'] and not r['period'][0]<=row['month']<=r['period'][1]:continue
        if r['settled_only'] and row['status']!='settled':continue
        v=row['gross']-(row['refund'] if r['net'] else 0)
        values.append(v*task.fx[row['currency']])
    if r['statistic']=='total':value=sum(values)
    elif not values:value=None
    elif r['statistic']=='average':value=sum(values)/len(values)
    elif r['statistic']=='largest':value=max(values)
    else:value=statistics.median(values)
    return {'value':value,'currency':'USD','statistic':r['statistic'],
            'region':r['region'],'period':r['period'],'settled_only':r['settled_only'],'net':r['net']}


def execute_workflow(task,selected,*,primary_available=None):
    sample=task.sample
    result={'success':False,'valid':False,'executed':[],'cost':0.,'artifact':None,'error':None,'failed_tool':None}
    plan=topological_plan(sample,frozenset(selected))
    if plan is None:result['error']='infeasible set or no legal execution order';return result
    result['valid']=True;rows=None;value=None;computed=None
    if primary_available is None:primary_available=not task.primary_fails
    try:
        for tool in plan:
            result['cost']+=sample.costs[tool]
            if tool in READERS:
                if tool=='read_primary' and not primary_available:
                    result['failed_tool']=tool;raise RuntimeError('primary source unavailable')
                rows=[dict(row,value=row['gross']) for row in task.ledger]
            elif tool=='filter_region':
                if task.request['region'] is not None:rows=[r for r in rows if r['region']==task.request['region']]
            elif tool=='filter_period':
                if task.request['period']:
                    a,b=task.request['period'];rows=[r for r in rows if a<=r['month']<=b]
            elif tool=='filter_settled':rows=[r for r in rows if r['status']=='settled']
            elif tool=='subtract_refunds':
                for row in rows:row['value']-=row['refund']
            elif tool=='convert_usd':
                for row in rows:row['value']*=task.fx[row['currency']]
            elif tool=='round_rows':
                for row in rows:row['value']=round(row['value'])
            elif tool in AGGREGATORS.values():
                vals=[r['value'] for r in rows]
                if tool=='sum_values':value=sum(vals);computed='total'
                elif tool=='mean_values':value=sum(vals)/len(vals) if vals else None;computed='average'
                elif tool=='max_values':value=max(vals) if vals else None;computed='largest'
                else:value=statistics.median(vals) if vals else None;computed='median'
            elif tool in REPORTS:
                report={**task.request,'value':value,'statistic':computed}
                if tool=='report_json':result['artifact']=json.dumps(report,sort_keys=True)
                else:
                    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(report));w.writeheader();w.writerow(report);result['artifact']=buf.getvalue()
            elif tool=='send_external_email':raise RuntimeError('External tool unavailable; no message sent')
            result['executed'].append(tool)
    except (RuntimeError,TypeError,KeyError,ValueError) as exc:
        result['error']=str(exc);return result
    if result['artifact'] is None:return result
    # Check machine-readable format as well as actual numeric content.
    if task.request['format']=='json':
        if 'report_json' not in selected:return result
        observed=json.loads(result['artifact'])
    else:
        if 'report_csv' not in selected:return result
        observed=next(csv.DictReader(io.StringIO(result['artifact'])))
        raw=observed['value'];observed['value']=float(raw) if raw else None
    expected=expected_report(task)
    observed_value=observed['value'];expected_value=expected['value']
    numeric=(observed_value is None and expected_value is None) or (
        isinstance(observed_value,(int,float)) and expected_value is not None and abs(observed_value-expected_value)<=1e-7)
    result['success']=numeric and observed['statistic']==expected['statistic']
    result['expected_value']=expected_value
    return result


def execute_with_read_recovery(task,selector):
    """At most one failed-read replan; cumulative tool budget includes failure."""
    chosen=selector(task.sample)
    first=execute_workflow(task,chosen)
    if first['failed_tool']!='read_primary':return {'attempts':[first],'success':first['success'],'total_cost':first['cost'],'replanned':False}
    remaining=task.sample.budget-first['cost']
    opts={k:v for k,v in task.sample.options.items() if k!='read_primary'}
    # A failed initial read has no completed data-transformation side effects.
    if first['executed']:raise RuntimeError('Recovery only supports initial source failure')
    anyof={k:[x for x in v if x in opts] for k,v in task.sample.requires_any.items() if k in opts}
    updated=replace(task.sample,options=opts,gold=frozenset(),budget=remaining,
        costs={k:v for k,v in task.sample.costs.items() if k in opts},
        excludes=[p for p in task.sample.excludes if set(p)<=set(opts)],requires_any=anyof,
        precedes={k:[x for x in v if x in opts] for k,v in task.sample.precedes.items() if k in opts},
        text=task.sample.text+'\nRuntime observation: primary read failed before any records were loaded. Primary is unavailable. Remaining budget: '+str(remaining))
    retry=execute_workflow(replace(task,sample=updated),selector(updated))
    total=first['cost']+retry['cost']
    return {'attempts':[first,retry],'success':retry['success'] and total<=task.sample.budget+1e-9,
            'total_cost':total,'replanned':True}
