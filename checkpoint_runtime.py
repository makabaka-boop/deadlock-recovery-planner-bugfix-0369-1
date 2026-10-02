"""Replay selected checkpoints against a stalled resource state."""
from deadlock_simulator import simulate


def apply_checkpoint(state, checkpoint):
    """Roll one active job back to one of its declared checkpoints.

    The job keeps only the held resources listed in ``keep``; every other
    held resource is released and must be re-acquired before the job may
    complete.  The job's original waiting demand is retained: it first
    continues waiting for that resource, then re-acquires the released
    resources one at a time in ascending resource id order.  Returns the
    rollback event.
    """
    job = state.jobs[checkpoint['job']]
    released = sorted(job.holding - set(checkpoint['keep']))
    for resource in released:
        state.resources[resource].holder = None
    job.holding = set(checkpoint['keep'])
    # The original wait (job.waiting_for) is left untouched; the released
    # resources are queued behind it and must all be granted again before
    # the job can complete.
    job.needs = list(released)
    if job.waiting_for is None and job.needs:
        job.waiting_for = job.needs.pop(0)
    return {'type': 'rollback', 'job': job.id, 'checkpoint': checkpoint['id'],
            'released': released, 'retained': sorted(job.holding)}


def replay(state, options):
    """Apply ``options`` (at most one checkpoint per job) on a clone of the
    stalled ``state`` and simulate to a stop.

    Returns ``(state, events, stuck)``: the mutated clone, the ordered
    rollback/grant/complete replay, and the jobs still deadlocked (empty
    when every job completed).
    """
    candidate = state.clone()
    events = []
    for option in options:
        events.append(apply_checkpoint(candidate, option))
    continuation, stuck = simulate(candidate)
    events.extend(continuation)
    return candidate, events, stuck
