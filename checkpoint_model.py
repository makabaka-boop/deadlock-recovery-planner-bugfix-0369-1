"""Checkpoint alternatives for the resource recovery API."""
from deadlock_simulator import build_state, _is_int

MAX_CHECKPOINTS = 20
REQUIRED_FIELDS = {'id', 'job', 'keep', 'cost'}


def parse_checkpoints(payload):
    """Validate the checkpoint declarations and build a fresh state.

    Returns ``(state, options)`` where ``options`` are the checkpoint dicts
    in declaration order.  Raises :class:`ValueError` on any inconsistency.
    """
    state = build_state(payload)
    raw_list = payload.get('checkpoints', [])
    if not isinstance(raw_list, list):
        raise ValueError("'checkpoints' must be a list")
    options = []
    seen = set()
    for raw in raw_list:
        if not isinstance(raw, dict) or set(raw) != REQUIRED_FIELDS:
            raise ValueError('checkpoint fields invalid')
        cid, jid, keep, cost = raw['id'], raw['job'], raw['keep'], raw['cost']
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('checkpoint id invalid')
        if not _is_int(jid) or jid not in state.jobs:
            raise ValueError('unknown checkpoint job')
        if (not isinstance(keep, list)
                or any(not _is_int(r) for r in keep)
                or len(set(keep)) != len(keep)):
            raise ValueError('keep must be unique')
        if not set(keep) <= state.jobs[jid].holding:
            raise ValueError('cannot retain unowned resource')
        if not _is_int(cost) or cost <= 0:
            raise ValueError('positive rollback cost required')
        seen.add(cid)
        options.append(dict(raw))
    if len(options) > MAX_CHECKPOINTS:
        raise ValueError('too many checkpoints')
    return state, options
