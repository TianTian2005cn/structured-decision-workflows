import hashlib
STAT={'total':'the total','average':'the average','largest':'the largest single','median':'the median'}
VERB=['Please compute','I need','Report','Calculate']
def nl(req,key):
    h=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    reg=' across all regions' if req['region'] is None else f" for the {req['region']} region"
    per=' over the whole ledger period' if not req['period'] else f" for the months {req['period'][0]} through {req['period'][1]}"
    st=', counting settled transactions only' if req['settled_only'] else ', counting transactions of every status'
    val='net (refund-adjusted)' if req['net'] else 'gross (before refunds)'
    return f"{VERB[h%4]} {STAT[req['statistic']]} {val} transaction value in USD{reg}{per}{st}, delivered as a {req['format'].upper()} report."
