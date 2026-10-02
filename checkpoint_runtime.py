"""Replay a selected checkpoint combination against a stalled resource state.

A rolled-back job keeps only the resources listed in the checkpoint's
``keep``; every other held resource is released and must be re-acquired
before the job may complete.  The job first re-issues its original waiting
request (if any), then re-acquires the released resources in ascending
resource id order, waiting for at most one resource at a time.  Only when
all of these demands are satisfied does the job become eligible for
completion, so every grant and release shows up in the replay.
"""
from deadlock_simulator import simulate


def apply_checkpoint(state, checkpoint):
    """Roll one job back to ``checkpoint`` and return the rollback event.

    The event records the job's full outstanding demand list (``awaiting``)
    so the following grants can be audited against the replay alone.
    """
    job = state.jobs[checkpoint['job']]
    released = sorted(job.holding - set(checkpoint['keep']))
    for resource in released:
        state.resources[resource].holder = None
    retained = sorted(checkpoint['keep'])
    job.holding = set(checkpoint['keep'])
    awaiting = ([job.waiting_for] if job.waiting_for is not None else []) + released
    job.waiting_for = awaiting[0] if awaiting else None
    job.pending = tuple(awaiting[1:])
    return {'type': 'rollback', 'job': job.id, 'checkpoint': checkpoint['id'],
            'released': released, 'retained': retained, 'awaiting': awaiting}


def replay(state, options):
    """Apply ``options`` (in order) to a clone of the stalled ``state`` and
    re-simulate.  Returns ``(final_state, events, stuck)``; ``state`` itself
    is left untouched."""
    candidate = state.clone()
    events = [apply_checkpoint(candidate, option) for option in options]
    continuation, stuck = simulate(candidate)
    events.extend(continuation)
    return candidate, events, stuck


def _feasible(state, plan):
    """Fast feasibility oracle for the solver's search loop: would rolling
    the stalled ``state`` back to ``plan`` let every job complete?

    Mirrors :func:`deadlock_simulator.simulate` exactly (same phase order,
    same grant adjudication) but builds no events and clones nothing;
    ``state`` is never mutated.  The reference :func:`replay` above still
    produces the winning plan's events.
    """
    holding = {jid: set(j.holding) for jid, j in state.jobs.items()}
    waiting = {jid: j.waiting_for for jid, j in state.jobs.items()}
    pending = {jid: j.pending for jid, j in state.jobs.items()}
    holder = {rid: r.holder for rid, r in state.resources.items()}
    active = set(state.jobs)
    for cp in plan:
        jid = cp['job']
        released = sorted(holding[jid] - set(cp['keep']))
        for rid in released:
            holder[rid] = None
        holding[jid] = set(cp['keep'])
        queue = ([waiting[jid]] if waiting[jid] is not None else []) + released
        waiting[jid] = queue[0] if queue else None
        pending[jid] = tuple(queue[1:])
    waiters = {}
    for jid in active:
        rid = waiting[jid]
        if rid is not None:
            waiters.setdefault(rid, set()).add(jid)
    while True:
        # Completion phase: every job whose demands are all met finishes.
        progressed = False
        for jid in sorted(active):
            if waiting[jid] is None:
                for rid in holding[jid]:
                    holder[rid] = None
                holding[jid] = set()
                active.discard(jid)
                progressed = True
        if progressed:
            continue
        # Grant phase: snapshot all grants, then apply them.
        grants = []
        for rid in sorted(holder):
            if holder[rid] is None and waiters.get(rid):
                grants.append((rid, min(waiters[rid])))
        if not grants:
            return not active
        for rid, jid in grants:
            holder[rid] = jid
            holding[jid].add(rid)
            waiters[rid].discard(jid)
            if pending[jid]:
                nxt = pending[jid][0]
                pending[jid] = pending[jid][1:]
                waiting[jid] = nxt
                waiters.setdefault(nxt, set()).add(jid)
            else:
                waiting[jid] = None
