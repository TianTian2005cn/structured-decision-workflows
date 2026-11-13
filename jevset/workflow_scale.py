"""V3 environments for a frozen V2 selector/controller; no algorithm tuning.

All environmental changes are explicit: longer ledgers, validation dependencies,
budget pressure, late faults, clarification goals and tool/schema transfer.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import csv,io,json,random,statistics
from .workflow_v2 import WorkflowTask,make_workflows,REGISTRY,AGGREGATORS,expected_report
from .evaluation import topological_plan
from .live import digest

@dataclass
class ScaleTask:
    base:WorkflowTask
    cohort:str
    cluster:str
    aliases:dict
    fault:str|None=None
    clarify:bool=False
    unit_scale:float=1.
    def signature(self):return digest({'state':self.base.sample.public_state(),'fault':self.fault,'clarify':self.clarify})


def make_scale_tasks(excluded_families,n_per=100,seed=2026103003):
    rng=random.Random(seed)
    candidates=[t for t in make_workflows(256,seed=seed) if digest(t.request) not in excluded_families]
    candidates.sort(key=lambda t:digest({'seed':seed,'request':t.request}))
    if len(candidates)<n_per:raise ValueError('Insufficient unreserved source families')
    selected=candidates[:n_per];out=[]
    for cohort in ['basic','medium','hard','transfer']:
        for i,t in enumerate(selected):
            t=replace(t,ledger=[dict(r) for r in t.ledger],request=dict(t.request))
            sample=replace(t.sample,options=dict(t.sample.options),requires={},
                requires_any={k:list(v) for k,v in t.sample.requires_any.items()},
                precedes={k:list(v) for k,v in t.sample.precedes.items()},
                excludes=[list(p) for p in t.sample.excludes],costs=dict(t.sample.costs))
            needed=set(sample.gold);fault=None;clarify=False;unit=1.
            # Failure independent of budget slack; infeasible recovery budgets
            # are retained and reported, rather than filtered out after results.
            if rng.random()<(.15 if cohort=='basic' else .22):fault='read_primary'
            if cohort!='basic':
                sample.options.update({
                    'validate_schema':'Validate required transaction fields and reject malformed records.',
                    'validate_fx':'Check that supplied offline FX rates cover every selected currency.',
                    'parse_dates':'Parse the month field before filtering a date period.',
                    'deduplicate_rows':'Remove duplicate transaction rows by transaction_id.',
                    'normalize_units':'Convert minor-unit gross and refund fields to major units before arithmetic.'})
                for op in ['validate_schema','validate_fx','parse_dates','deduplicate_rows','normalize_units']:
                    sample.requires_any[op]=['read_primary','read_backup'];sample.costs[op]=1.
                sample.requires.update({'parse_dates':['validate_schema'],'deduplicate_rows':['validate_schema'],
                                        'validate_fx':['validate_schema'],'normalize_units':['validate_schema']})
                sample.requires['filter_period']=['parse_dates']
                sample.requires['convert_usd']=['validate_fx']
                sample.precedes['deduplicate_rows']=list(AGGREGATORS.values())+['subtract_refunds','convert_usd']
                sample.precedes['normalize_units']=['subtract_refunds','convert_usd']+list(AGGREGATORS.values())
                sample.precedes['validate_schema']=['filter_region','filter_period','filter_settled']
                # Real duplicate rows, with known stable transaction IDs.
                for j,row in enumerate(t.ledger):row['transaction_id']=f'r{j}'
                if i%2==0:
                    t.ledger.extend(dict(r) for r in t.ledger[:4]);needed.add('deduplicate_rows');needed.add('validate_schema')
                if t.request['period']:needed.update(['parse_dates','validate_schema'])
                if 'convert_usd' in needed:needed.update(['validate_fx','validate_schema'])
                if cohort in {'hard','transfer'} and i%3==0:
                    unit=100.
                    for row in t.ledger:row['gross']*=unit;row['refund']*=unit
                    needed.update(['normalize_units','validate_schema'])
                if cohort=='hard':
                    sample.options['clarify_scope']='Request clarification when the goal omits whether revenue means gross or net; produce no numeric report.'
                    sample.costs['clarify_scope']=1.
                    if i%10==0:
                        clarify=True;t.request['net']=None;needed={'clarify_scope'};fault=None
                    elif i%7==0:
                        fault='convert_usd' if 'convert_usd' in needed else AGGREGATORS[t.request['statistic']]
                if cohort=='transfer':
                    sample.options['count_rows']='Count records; does not compute the requested revenue or value statistic.'
                    sample.costs['count_rows']=1.;sample.requires_any['count_rows']=['read_primary','read_backup']
            # Keep frozen reader IDs because the unchanged V2 controller uses
            # these explicit resource IDs; transfer all other tool IDs/schema.
            aliases={k:k for k in sample.options}
            if cohort=='transfer':
                aliases={k:('new_'+k if k not in {'read_primary','read_backup'} else k) for k in sample.options}
                replacements={'transaction':'inventory movement','revenue':'inventory value','refund':'damaged-stock loss',
                              'region':'warehouse','settled':'confirmed','ledger':'stock ledger'}
                descriptions={}
                for k,v in sample.options.items():
                    for a,b in replacements.items():v=v.replace(a,b)
                    descriptions[aliases[k]]=v
                sample.options=descriptions
                sample.requires={aliases[k]:[aliases[v] for v in vs] for k,vs in sample.requires.items()}
                sample.requires_any={aliases[k]:[aliases[v] for v in vs] for k,vs in sample.requires_any.items()}
                sample.precedes={aliases[k]:[aliases[v] for v in vs] for k,vs in sample.precedes.items()}
                sample.excludes=[[aliases[v] for v in p] for p in sample.excludes]
                sample.costs={aliases[k]:v for k,v in sample.costs.items()}
            # No budget adjustment using failure outcomes or hidden fault type.
            slack=[0,1,2,3][i%4] if cohort!='basic' else [1,2,3][i%3]
            sample.budget=sum(sample.costs[aliases[k]] for k in needed)+slack
            sample.gold=frozenset(aliases[k] for k in needed)
            sample.id=f'scale-v3-{cohort}-{i:03d}';sample.split='development'
            if cohort=='transfer':
                visible_request={('warehouse' if k=='region' else 'net_after_damage' if k=='net' else k):v for k,v in t.request.items()}
                visible_rows=[{('warehouse' if k=='region' else 'inbound_value' if k=='gross' else 'damage_loss' if k=='refund' else k):v for k,v in r.items()} for r in t.ledger]
                schema_note='Inventory schema: warehouse corresponds to location; inbound_value is gross value, damage_loss is a deduction. '
            else:visible_request=t.request;visible_rows=t.ledger;schema_note=''
            text=('Produce exactly the requested JSON or CSV numeric report. Never round individual rows or send external messages. '
                  'If net/gross is unspecified, request clarification instead of guessing. Duplicated transaction IDs represent the same transaction. '
                  f'Amount unit scale is {unit:g}: divide raw amounts by this scale before arithmetic. '+schema_note+
                  '\nRequest: '+json.dumps(visible_request)+'\nOffline FX rates to USD: '+json.dumps(t.fx)+
                  '\nTransactions: '+json.dumps(visible_rows))
            sample.text=text
            t=replace(t,sample=sample,primary_fails=fault=='read_primary',difficulty=cohort)
            out.append(ScaleTask(t,cohort,digest({'base_request':{**t.request,'net':False if clarify else t.request['net']},'index':i}),aliases,fault,clarify,unit))
    return out


def scale_oracle(task):
    if task.clarify:return {'requires_clarification':True}
    # Independent reference: normalize and deduplicate original records, then
    # evaluate the request directly; no selected tools or gold set are read.
    rows=[];seen=set()
    for row in task.base.ledger:
        rid=row.get('transaction_id')
        if rid is not None and rid in seen:continue
        if rid is not None:seen.add(rid)
        rows.append({**row,'gross':row['gross']/task.unit_scale,'refund':row['refund']/task.unit_scale})
    return expected_report(replace(task.base,ledger=rows))


def execute_scale(task,selected):
    sample=task.base.sample;selected=frozenset(selected)
    result={'success':False,'valid':False,'executed':[],'cost':0.,'artifact':None,'error':None,'failed_tool':None}
    plan=topological_plan(sample,selected)
    if plan is None:result['error']='infeasible set or no execution order';return result
    result['valid']=True;inverse={v:k for k,v in task.aliases.items()};rows=None;value=None;computed=None
    try:
        for public in plan:
            op=inverse[public];result['cost']+=sample.costs[public]
            if op==task.fault:
                result['failed_tool']=public;raise RuntimeError('injected failure: '+op)
            if op in {'read_primary','read_backup'}:rows=[dict(r,value=r['gross']) for r in task.base.ledger]
            elif op=='validate_schema':
                if any(not {'region','month','status','currency','gross','refund'}<=r.keys() for r in rows):raise RuntimeError('schema failure')
            elif op=='validate_fx':
                if any(r['currency'] not in task.base.fx for r in rows):raise RuntimeError('missing FX rate')
            elif op=='parse_dates':
                if any(len(r['month'])!=7 or r['month'][4]!='-' for r in rows):raise RuntimeError('invalid month')
            elif op=='deduplicate_rows':
                seen=set();clean=[]
                for row in rows:
                    rid=row.get('transaction_id')
                    if rid is None or rid not in seen:clean.append(row)
                    seen.add(rid)
                rows=clean
            elif op=='normalize_units':
                for row in rows:row['value']/=task.unit_scale;row['refund']/=task.unit_scale
            elif op=='filter_region':
                r=task.base.request['region']
                if r is not None:rows=[x for x in rows if x['region']==r]
            elif op=='filter_period':
                period=task.base.request['period']
                if period:rows=[x for x in rows if period[0]<=x['month']<=period[1]]
            elif op=='filter_settled':rows=[x for x in rows if x['status']=='settled']
            elif op=='subtract_refunds':
                for row in rows:row['value']-=row['refund']
            elif op=='convert_usd':
                for row in rows:row['value']*=task.base.fx[row['currency']]
            elif op=='round_rows':
                for row in rows:row['value']=round(row['value'])
            elif op in AGGREGATORS.values():
                vals=[r['value'] for r in rows];computed=next(k for k,v in AGGREGATORS.items() if v==op)
                if op=='sum_values':value=sum(vals)
                elif not vals:value=None
                elif op=='mean_values':value=sum(vals)/len(vals)
                elif op=='max_values':value=max(vals)
                else:value=statistics.median(vals)
            elif op in {'report_json','report_csv'}:
                r={**task.base.request,'statistic':computed,'value':value}
                if op=='report_json':result['artifact']=json.dumps(r)
                else:
                    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(r));w.writeheader();w.writerow(r);result['artifact']=buf.getvalue()
            elif op=='clarify_scope':result['artifact']='REQUEST_CLARIFICATION'
            elif op=='send_external_email':raise RuntimeError('external tool unavailable')
            elif op=='count_rows':value=len(rows);computed='count'
            elif op=='plot_preview':pass # distractor; no final numeric report
            else:raise RuntimeError('unknown tool contract')
            result['executed'].append(public)
    except (RuntimeError,ValueError,KeyError,TypeError) as e:result['error']=str(e);return result
    expected=scale_oracle(task)
    if task.clarify:result['success']=result['artifact']=='REQUEST_CLARIFICATION';return result
    if result['artifact'] is None or result['artifact']=='REQUEST_CLARIFICATION':return result
    report_tool=task.aliases['report_'+task.base.request['format']]
    if report_tool not in selected:return result
    if task.base.request['format']=='json':observed=json.loads(result['artifact'])
    else:
        observed=next(csv.DictReader(io.StringIO(result['artifact'])));observed['value']=float(observed['value']) if observed['value'] else None
    observed_value=observed['value'];target=expected['value']
    result['success']=observed['statistic']==expected['statistic'] and ((target is None and observed_value is None) or (
        target is not None and isinstance(observed_value,(int,float)) and abs(observed_value-target)<1e-7))
    result['expected_value']=target
    return result


def frozen_read_recovery(task,selector):
    """Exact V2 one-initial-read-replan control flow, new executor adapter only."""
    first=execute_scale(task,selector(task.base.sample))
    if first['failed_tool']!='read_primary':return {'attempts':[first],'success':first['success'],'total_cost':first['cost'],'replanned':False}
    if first['executed']:raise RuntimeError('V2 recovery supports initial read only')
    s=task.base.sample;remaining=s.budget-first['cost'];opts={k:v for k,v in s.options.items() if k!='read_primary'}
    updated=replace(s,options=opts,gold=frozenset(),budget=remaining,costs={k:v for k,v in s.costs.items() if k in opts},
        excludes=[p for p in s.excludes if set(p)<=set(opts)],
        requires={k:[x for x in v if x in opts] for k,v in s.requires.items() if k in opts},
        requires_any={k:[x for x in v if x in opts] for k,v in s.requires_any.items() if k in opts},
        precedes={k:[x for x in v if x in opts] for k,v in s.precedes.items() if k in opts},
        text=s.text+'\nRuntime observation: primary read failed before any records were loaded. Primary is unavailable. Remaining budget: '+str(remaining))
    retry=execute_scale(replace(task,base=replace(task.base,sample=updated)),selector(updated));total=first['cost']+retry['cost']
    return {'attempts':[first,retry],'success':retry['success'] and total<=s.budget+1e-9,'total_cost':total,'replanned':True}
