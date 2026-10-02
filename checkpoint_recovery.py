"""Recovery through declared checkpoints, without aborting jobs."""
from itertools import combinations, product

from deadlock_simulator import simulate
from checkpoint_model import parse_checkpoints
from checkpoint_runtime import _feasible, replay

STATUS_COMPLETED = 'completed'
STATUS_RECOVERED = 'recovered'
STATUS_UNRESOLVABLE = 'unresolvable'


def _ranked_plans(by_job):
    """Every non-empty rollback plan with at most one checkpoint per job,
    as ``((cost, sorted ids), plan)`` pairs sorted by the optimality key:
    total rollback cost first, then the sorted checkpoint id list.  Each
    plan is a tuple ordered by ascending job id so replays are
    deterministic."""
    plans = []
    jobs = sorted(by_job)
    for size in range(1, len(jobs) + 1):
        for job_subset in combinations(jobs, size):
            for plan in product(*(by_job[j] for j in job_subset)):
                key = (sum(o['cost'] for o in plan),
                       sorted(o['id'] for o in plan))
                plans.append((key, plan))
    plans.sort(key=lambda ranked: ranked[0])
    return plans


def solve_checkpoints(payload):
    """Validate, simulate, and resolve a deadlock by checkpoint rollback.

    Every original job must complete normally; no job is ever aborted.
    Plans are considered in ascending order of the optimality key, so the
    first feasible plan found is the minimum-cost one, with ties broken by
    the lexicographically smallest sorted checkpoint id list.  When no plan
    is feasible, no rollback is performed at all.
    """
    state, options = parse_checkpoints(payload)
    prefix, stuck = simulate(state)
    if not stuck:
        return {'status': STATUS_COMPLETED, 'events': prefix, 'checkpoints': [], 'cost': 0}

    # Any stuck job may roll back to one of its own declared checkpoints;
    # protected jobs cannot be aborted but use checkpoints like any other.
    by_job = {}
    for option in options:
        if option['job'] in stuck:
            by_job.setdefault(option['job'], []).append(option)

    best = None
    for key, plan in _ranked_plans(by_job):
        if _feasible(state, plan):
            best = (key, plan)
            break
    if best is None:
        return {
            'status': STATUS_UNRESOLVABLE,
            'events': prefix,
            'stuck': stuck,
            'message': (
                'Deadlock cannot be resolved: no combination of the declared '
                f'checkpoints lets every stuck job {stuck} complete. '
                'No rollback was performed.'
            ),
        }
    (cost, checkpoint_ids), plan = best
    _, events, left = replay(state, plan)
    assert not left, "chosen checkpoint plan must resolve the deadlock"
    return {
        'status': STATUS_RECOVERED,
        'events': prefix + events,
        'cost': cost,
        'checkpoints': checkpoint_ids,
    }
