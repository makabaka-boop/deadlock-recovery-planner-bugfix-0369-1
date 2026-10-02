"""Replay a selected checkpoint against a stalled resource state."""
from deadlock_simulator import simulate

def apply_checkpoint(state, checkpoint):
    job=state.jobs[checkpoint['job']]
    released=sorted(job.holding-set(checkpoint['keep']))
    for resource in released:state.resources[resource].holder=None
    job.holding=set(checkpoint['keep'])
    job.waiting_for=None
    return {'type':'rollback','job':job.id,'checkpoint':checkpoint['id'],'released':released,
            'retained':sorted(job.holding)}

def replay(state, options):
    candidate=state.clone();events=[]
    for option in options:events.append(apply_checkpoint(candidate,option))
    continuation,stuck=simulate(candidate)
    events.extend(continuation)
    return candidate,events,stuck
