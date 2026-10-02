"""Recovery through declared checkpoints, without aborting jobs."""
from itertools import combinations
from deadlock_simulator import simulate
from checkpoint_model import parse_checkpoints
from checkpoint_runtime import replay

def solve_checkpoints(payload):
    state,options=parse_checkpoints(payload)
    prefix,stuck=simulate(state)
    if not stuck:return {'status':'completed','events':prefix,'checkpoints':[],'cost':0}
    available=[o for o in options if o['job'] in stuck and state.jobs[o['job']].abortable]
    byjob={}
    for o in sorted(available,key=lambda o:(o['cost'],o['id'])):byjob.setdefault(o['job'],o)
    choices=list(byjob.values());best=None
    for size in range(1,len(choices)+1):
        for subset in combinations(choices,size):
            final,events,left=replay(state,subset)
            if left:continue
            key=(sum(o['cost'] for o in subset),sorted(o['id'] for o in subset))
            if best is None or key<best[0]:best=(key,events)
    if best is None:return {'status':'unresolvable','events':prefix,'stuck':stuck}
    return {'status':'recovered','events':prefix+best[1],'cost':best[0][0],'checkpoints':best[0][1]}
