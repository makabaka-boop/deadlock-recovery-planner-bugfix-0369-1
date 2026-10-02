"""Recovery through declared checkpoints, without aborting jobs."""
from itertools import combinations, product
from deadlock_simulator import simulate
from checkpoint_model import parse_checkpoints
from checkpoint_runtime import replay


def solve_checkpoints(payload):
    """Validate the payload, simulate, then resolve any deadlock by rolling
    jobs back to their declared checkpoints.

    Every stuck job may use at most one of its own checkpoints; protected
    jobs cannot be aborted but may roll back like any other job.  Among the
    combinations that let every original job complete, the one with minimum
    total rollback cost wins, ties broken by the lexicographically smallest
    sorted checkpoint id list.  When no legal combination exists, no partial
    recovery plan is returned.
    """
    state, options = parse_checkpoints(payload)
    prefix, stuck = simulate(state)
    if not stuck:
        return {'status': 'completed', 'events': prefix, 'checkpoints': [], 'cost': 0}

    # Group every checkpoint of every stuck job; protected jobs included.
    by_job = {}
    for option in options:
        if option['job'] in stuck:
            by_job.setdefault(option['job'], []).append(option)

    best = None
    jobs = sorted(by_job)
    for size in range(1, len(jobs) + 1):
        for job_subset in combinations(jobs, size):
            for subset in product(*(by_job[j] for j in job_subset)):
                key = (sum(o['cost'] for o in subset),
                       sorted(o['id'] for o in subset))
                if best is not None and key >= best[0]:
                    continue  # cannot improve on the current best
                _, events, left = replay(state, subset)
                if left:
                    continue  # not every job completes: illegal combination
                best = (key, events)
    if best is None:
        return {'status': 'unresolvable', 'events': prefix, 'stuck': stuck}
    return {'status': 'recovered', 'events': prefix + best[1],
            'cost': best[0][0], 'checkpoints': best[0][1]}
