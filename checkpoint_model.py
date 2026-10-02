"""Checkpoint alternatives for the resource recovery API."""
from deadlock_simulator import build_state

def parse_checkpoints(payload):
    state=build_state(payload)
    options=[]
    seen=set()
    for raw in payload.get('checkpoints',[]):
        if set(raw)!={'id','job','keep','cost'}:raise ValueError('checkpoint fields invalid')
        cid,jid,keep,cost=raw['id'],raw['job'],raw['keep'],raw['cost']
        if not isinstance(cid,str) or not cid or cid in seen:raise ValueError('checkpoint id invalid')
        if jid not in state.jobs:raise ValueError('unknown checkpoint job')
        if not isinstance(keep,list) or len(set(keep))!=len(keep):raise ValueError('keep must be unique')
        if not set(keep)<=state.jobs[jid].holding:raise ValueError('cannot retain unowned resource')
        if type(cost) is not int or cost<=0:raise ValueError('positive rollback cost required')
        seen.add(cid);options.append(dict(raw))
    if len(options)>20:raise ValueError('too many checkpoints')
    return state,options
