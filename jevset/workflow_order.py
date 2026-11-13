"""Second closed workflow domain (paper 3 revision): order fulfilment and invoicing.

Differs from the ledger-report domain in its decision structure: most decisions are 1-of-n exclusive choices
driven by business rules stated in the instruction (discount policy, shipping mode, tax regime), rather than
optional filters. All decisions are closed by construction: arguments are bound from the request (K1), admissibility
is declared (requires-any, precedence, exclusions, costs, budget; K2) and an initial source failure is observable
(K3). The oracle recomputes the invoice total from the raw order lines and the request, independently of the tool
registry, the declared constraints and Sample.gold.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import json, random
from .schema import Sample
from .evaluation import topological_plan

REGISTRY = {
 'load_order': 'Load the order lines from the primary order service. It may fail at runtime.',
 'load_order_archive': 'Load an identical copy of the order lines from the archive; costs two units.',
 'check_stock': 'Verify stock availability for all order lines (required before any shipment is booked).',
 'apply_coupon': 'Apply the customer coupon to the merchandise subtotal.',
 'apply_loyalty_discount': 'Apply the loyalty discount of the customer tier to the merchandise subtotal.',
 'ship_ground': 'Book ground shipping for the order and add its fee.',
 'ship_express': 'Book express shipping for the order and add its fee.',
 'ship_freight': 'Book international freight shipping for the order and add its fee.',
 'add_gift_wrap': 'Add gift wrapping to the order and its fee.',
 'tax_domestic': 'Compute domestic sales tax.',
 'tax_eu_vat': 'Compute EU value-added tax.',
 'tax_export_exempt': 'Mark the order as tax-exempt export (zero tax).',
 'compute_total': 'Compute the invoice total from all current amounts.',
 'invoice_pdf': 'Issue the invoice as a PDF document record.',
 'invoice_edi': 'Issue the invoice as an EDI message record.',
 'price_override': 'Manually override unit prices. Not permitted without a manager request.',
 'send_marketing_email': 'Send a marketing e-mail to the customer. Not permitted in this workflow.'}
LOADERS = ['load_order', 'load_order_archive']
DISC = ['apply_coupon', 'apply_loyalty_discount']
SHIP = ['ship_ground', 'ship_express', 'ship_freight']
TAX = ['tax_domestic', 'tax_eu_vat', 'tax_export_exempt']
INV = ['invoice_pdf', 'invoice_edi']
POLICY = ('Business rules: a coupon gives 10% off the merchandise subtotal; customers of tier "gold" get a 5% loyalty '
          'discount; discounts never stack, and a coupon, if present, takes precedence over the loyalty discount. '
          'International orders always ship by freight (40 + 2 per kg); otherwise express (25 + 1.5 per kg) if requested, '
          'else ground (8 + 0.5 per kg). Gift wrapping costs 5 per order. Domestic orders pay 8% sales tax on discounted '
          'merchandise and gift wrapping; EU orders pay 20% VAT on discounted merchandise, gift wrapping and shipping; '
          'international orders are tax-exempt exports. The invoice total is discounted merchandise + gift wrapping + '
          'shipping + tax. Do not override prices or send marketing messages. Each tool runs at most once; use the '
          'archive only when necessary.')

@dataclass
class OrderTask:
    sample: Sample
    lines: list[dict]
    request: dict
    primary_fails: bool = False
    difficulty: str = 'standard'

def gold_tools(req):
    g = {'load_order', 'check_stock', 'compute_total', 'invoice_' + req['invoice_format']}
    if req['coupon']: g.add('apply_coupon')
    elif req['customer_tier'] == 'gold': g.add('apply_loyalty_discount')
    g.add('ship_freight' if req['destination'] == 'international' else 'ship_express' if req['express'] else 'ship_ground')
    g.add({'domestic': 'tax_domestic', 'eu': 'tax_eu_vat', 'international': 'tax_export_exempt'}[req['destination']])
    if req['gift_wrap']: g.add('add_gift_wrap')
    return g

def make_orders(n=96, seed=20261107, split='test'):
    rng = random.Random(seed); out = []
    for i in range(n):
        req = {'customer_tier': ['standard', 'gold'][(i // 2) % 2], 'destination': ['domestic', 'eu', 'international'][i % 3],
               'coupon': 'SAVE10' if (i // 4) % 2 else None, 'express': bool((i // 8) % 2), 'gift_wrap': bool((i // 16) % 2),
               'invoice_format': ['pdf', 'edi'][(i // 32) % 2], 'currency': 'USD'}
        lines = [{'sku': f'SKU-{rng.randint(100, 999)}', 'qty': rng.randint(1, 5), 'unit_price': rng.randint(500, 9000) / 100,
                  'weight_kg': rng.randint(1, 40) / 10} for _ in range(rng.randint(3, 8))]
        gold = gold_tools(req)
        anyof = {k: LOADERS for k in ['check_stock', 'price_override'] + DISC + ['add_gift_wrap']}
        anyof.update({k: ['check_stock'] for k in SHIP})
        anyof.update({k: SHIP for k in TAX})
        anyof['compute_total'] = TAX
        anyof.update({k: ['compute_total'] for k in INV})
        anyof['send_marketing_email'] = INV
        prec = {d: TAX + ['compute_total'] for d in DISC}
        prec['add_gift_wrap'] = TAX + ['compute_total']
        prec['price_override'] = DISC + TAX + ['compute_total']
        excl = [LOADERS, DISC, INV] + [[a, b] for x, a in enumerate(SHIP) for b in SHIP[x + 1:]] + [[a, b] for x, a in enumerate(TAX) for b in TAX[x + 1:]]
        costs = {k: 1. for k in REGISTRY}; costs['load_order_archive'] = 2.; costs['send_marketing_email'] = 20.; costs['price_override'] = 3.
        fail = (i % 6 == 5)
        budget = len(gold) + (2 if fail else (i % 3))
        text = ('Process this customer order and issue exactly one invoice. ' + POLICY + '\nRequest: ' + json.dumps(req) +
                '\nOrder lines: ' + json.dumps(lines))
        s = Sample(f'order-{seed}-{i:03d}', 'tools', text, REGISTRY.copy(), frozenset(gold), split=split,
                   requires_any=anyof, precedes=prec, excludes=excl, costs=costs, budget=budget)
        out.append(OrderTask(s, lines, req, fail, 'read_failure' if fail else 'tight_budget' if i % 3 == 0 else 'distractors'))
    return out

def expected_total(t: OrderTask):
    r = t.request
    merch = sum(l['qty'] * l['unit_price'] for l in t.lines); kg = sum(l['qty'] * l['weight_kg'] for l in t.lines)
    if r['coupon']: merch *= .9
    elif r['customer_tier'] == 'gold': merch *= .95
    ship = 40 + 2 * kg if r['destination'] == 'international' else (25 + 1.5 * kg if r['express'] else 8 + .5 * kg)
    gift = 5. if r['gift_wrap'] else 0.
    tax = {'domestic': .08 * (merch + gift), 'eu': .2 * (merch + gift + ship), 'international': 0.}[r['destination']]
    return merch + gift + ship + tax

def execute_order(task: OrderTask, selected, *, primary_available=None):
    s = task.sample
    res = {'success': False, 'valid': False, 'executed': [], 'cost': 0., 'artifact': None, 'error': None, 'failed_tool': None}
    plan = topological_plan(s, frozenset(selected))
    if plan is None: res['error'] = 'infeasible set or no legal execution order'; return res
    res['valid'] = True
    if primary_available is None: primary_available = not task.primary_fails
    st = {}
    try:
        for tool in plan:
            res['cost'] += s.costs[tool]
            if tool in LOADERS:
                if tool == 'load_order' and not primary_available: res['failed_tool'] = tool; raise RuntimeError('primary order service unavailable')
                st['lines'] = [dict(l) for l in task.lines]
                st['merch'] = sum(l['qty'] * l['unit_price'] for l in st['lines']); st['kg'] = sum(l['qty'] * l['weight_kg'] for l in st['lines'])
            elif tool == 'check_stock': st['stock'] = True
            elif tool == 'apply_coupon':
                if not task.request['coupon']: raise RuntimeError('no coupon on this order')
                st['merch'] *= .9
            elif tool == 'apply_loyalty_discount':
                st['merch'] *= .95 if task.request['customer_tier'] == 'gold' else 1.
            elif tool == 'ship_ground': st['ship'] = 8 + .5 * st['kg']
            elif tool == 'ship_express': st['ship'] = 25 + 1.5 * st['kg']
            elif tool == 'ship_freight': st['ship'] = 40 + 2 * st['kg']
            elif tool == 'add_gift_wrap': st['gift'] = 5.
            elif tool == 'tax_domestic': st['tax'] = .08 * (st['merch'] + st.get('gift', 0.))
            elif tool == 'tax_eu_vat': st['tax'] = .2 * (st['merch'] + st.get('gift', 0.) + st['ship'])
            elif tool == 'tax_export_exempt': st['tax'] = 0.
            elif tool == 'compute_total': st['total'] = st['merch'] + st.get('gift', 0.) + st['ship'] + st['tax']
            elif tool in INV: res['artifact'] = {'format': tool.split('_')[1], 'total': st['total']}
            elif tool == 'price_override':
                for l in st['lines']: l['unit_price'] = round(l['unit_price'])
                st['merch'] = sum(l['qty'] * l['unit_price'] for l in st['lines'])
            elif tool == 'send_marketing_email': raise RuntimeError('marketing e-mail not permitted; nothing sent')
            res['executed'].append(tool)
    except (RuntimeError, KeyError, TypeError) as exc:
        res['error'] = str(exc); return res
    a = res['artifact']
    if a is None: return res
    res['expected_total'] = expected_total(task)
    res['success'] = a['format'] == task.request['invoice_format'] and abs(a['total'] - res['expected_total']) <= 1e-6
    return res

def execute_order_with_recovery(task, selector):
    chosen = selector(task.sample); first = execute_order(task, chosen)
    if first['failed_tool'] != 'load_order': return {'attempts': [first], 'success': first['success'], 'total_cost': first['cost'], 'replanned': False}
    s = task.sample; remaining = s.budget - first['cost']
    opts = {k: v for k, v in s.options.items() if k != 'load_order'}
    upd = replace(s, options=opts, gold=frozenset(), budget=remaining, costs={k: v for k, v in s.costs.items() if k in opts},
                  excludes=[p for p in s.excludes if set(p) <= set(opts)],
                  requires_any={k: [x for x in v if x in opts] for k, v in s.requires_any.items() if k in opts},
                  precedes={k: [x for x in v if x in opts] for k, v in s.precedes.items() if k in opts},
                  text=s.text + '\nRuntime observation: the primary order service failed before any order lines were loaded. It is unavailable. Remaining budget: ' + str(remaining))
    retry = execute_order(replace(task, sample=upd), selector(upd)); total = first['cost'] + retry['cost']
    return {'attempts': [first, retry], 'success': retry['success'] and total <= s.budget + 1e-9, 'total_cost': total, 'replanned': True}
