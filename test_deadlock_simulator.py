"""Tests for the deadlock simulator backend.

The cross-check test follows the required methodology: it enumerates
small abort subsets, re-simulates each one independently, and verifies
both that the returned abort set resolves the deadlock and that no
cheaper (or lexicographically smaller) set would.  A separate replay
verifier re-applies the returned event stream from scratch without
using any simulator internals.
"""

import copy
import http.client
import itertools
import json
import random
import threading
import unittest

from deadlock_simulator import (
    MAX_JOBS,
    MAX_RESOURCES,
    abort_job,
    build_state,
    find_min_abort_set,
    simulate,
    solve,
)
from backend_server import create_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mk_payload(job_specs, extra_resources=()):
    """Build a consistent payload; resource holders are derived from the
    jobs' ``holding`` lists, resources only mentioned in ``waiting_for``
    (or ``extra_resources``) start out free."""
    resources = {}
    for spec in job_specs:
        for r in spec.get("holding", []):
            resources[r] = spec["id"]
        w = spec.get("waiting_for")
        if w is not None:
            resources.setdefault(w, None)
    for r in extra_resources:
        resources.setdefault(r, None)
    return {
        "jobs": job_specs,
        "resources": [{"id": r, "holder": h} for r, h in sorted(resources.items())],
    }


def replay_events(payload, events):
    """Independently re-apply an event replay to the input state.

    Shares no code with the simulator: every event is checked for
    legality against a locally maintained state.  Returns the final
    ``(active, waiting, holder)`` triple.
    """
    holding = {j["id"]: set(j.get("holding", [])) for j in payload["jobs"]}
    waiting = {j["id"]: j.get("waiting_for") for j in payload["jobs"]}
    abortable = {j["id"]: j.get("abortable", False) for j in payload["jobs"]}
    holder = {r["id"]: r.get("holder") for r in payload["resources"]}
    active = set(holding)
    checked_stall = False
    for ev in events:
        kind = ev["type"]
        if kind == "grant":
            j, r = ev["job"], ev["resource"]
            assert j in active, f"grant to inactive job {j}"
            assert waiting[j] == r, f"job {j} is not waiting for resource {r}"
            assert holder[r] is None, f"resource {r} is not idle"
            holder[r] = j
            holding[j].add(r)
            waiting[j] = None
        elif kind == "complete":
            j = ev["job"]
            assert j in active, f"complete event for inactive job {j}"
            assert waiting[j] is None, f"job {j} completes while still waiting"
            assert sorted(holding[j]) == ev["released"], f"job {j} released set mismatch"
            for r in holding[j]:
                assert holder[r] == j, f"resource {r} not held by job {j}"
                holder[r] = None
            holding[j] = set()
            active.discard(j)
        elif kind == "abort":
            j = ev["job"]
            if not checked_stall:
                # Aborts may only start once the system is truly stuck:
                # every active job must be waiting for a held resource.
                for a in active:
                    assert waiting[a] is not None, f"job {a} could still complete"
                    assert holder[waiting[a]] is not None, (
                        f"job {a} could still be granted resource {waiting[a]}")
                checked_stall = True
            assert j in active, f"abort event for inactive job {j}"
            assert abortable[j], f"protected job {j} was aborted"
            assert sorted(holding[j]) == ev["released"], f"job {j} released set mismatch"
            for r in holding[j]:
                assert holder[r] == j, f"resource {r} not held by job {j}"
                holder[r] = None
            holding[j] = set()
            active.discard(j)
        else:
            raise AssertionError(f"unknown event type {kind!r}")
    return active, waiting, holder


def brute_force_min_abort_set(payload):
    """Enumerate every abortable stuck subset and independently re-simulate
    each candidate; return the minimum (cost, ids) feasible set or None."""
    state = build_state(payload)
    _, stuck = simulate(state)
    abortable = [j for j in stuck if state.jobs[j].abortable]
    best_key = None
    best = None
    for size in range(len(abortable) + 1):
        for combo in itertools.combinations(abortable, size):
            candidate = state.clone()
            for jid in combo:
                abort_job(candidate, jid)
            _, remaining = simulate(candidate)
            if not remaining:
                ids = sorted(combo)
                key = (sum(state.jobs[j].abort_cost for j in ids), ids)
                if best_key is None or key < best_key:
                    best_key, best = key, ids
    return best


def random_payload(rng):
    n_jobs = rng.randint(1, 8)
    n_res = rng.randint(1, 15)
    job_ids = rng.sample(range(1, 40), n_jobs)
    res_ids = rng.sample(range(100, 200), n_res)
    holder = {r: (rng.choice(job_ids) if rng.random() < 0.7 else None) for r in res_ids}
    jobs = []
    for jid in job_ids:
        holding = [r for r in res_ids if holder[r] == jid]
        choices = [r for r in res_ids if r not in holding]
        waiting_for = rng.choice(choices) if choices and rng.random() < 0.6 else None
        abortable = rng.random() < 0.6
        job = {"id": jid, "holding": holding, "waiting_for": waiting_for,
               "abortable": abortable}
        if abortable:
            job["abort_cost"] = rng.randint(1, 9)
        jobs.append(job)
    resources = [{"id": r, "holder": holder[r]} for r in res_ids]
    return {"jobs": jobs, "resources": resources}


# ---------------------------------------------------------------------------
# Simulation semantics
# ---------------------------------------------------------------------------

class SimulationTests(unittest.TestCase):
    def test_idle_jobs_complete_first_then_grants_by_smallest_id(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": True, "abort_cost": 4},
            {"id": 2, "holding": [], "waiting_for": None, "abortable": True, "abort_cost": 1},
            {"id": 3, "holding": [3], "waiting_for": 2, "abortable": True, "abort_cost": 1},
        ])
        result = solve(payload)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["events"], [
            {"type": "complete", "job": 2, "released": []},
            {"type": "grant", "job": 1, "resource": 2},   # smallest-id waiter of r2
            {"type": "complete", "job": 1, "released": [1, 2]},
            {"type": "grant", "job": 3, "resource": 2},
            {"type": "complete", "job": 3, "released": [2, 3]},
        ])
        self.assertEqual(result["aborted"], [])

    def test_completion_frees_resource_for_waiter(self):
        payload = mk_payload([
            {"id": 1, "holding": [], "waiting_for": 1, "abortable": False},
            {"id": 2, "holding": [1], "waiting_for": None, "abortable": False},
        ])
        result = solve(payload)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["events"], [
            {"type": "complete", "job": 2, "released": [1]},
            {"type": "grant", "job": 1, "resource": 1},
            {"type": "complete", "job": 1, "released": [1]},
        ])

    def test_empty_system_completes_immediately(self):
        result = solve({"jobs": [], "resources": []})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["events"], [])

    def test_free_resource_nobody_waits_for_stays_idle(self):
        payload = mk_payload(
            [{"id": 5, "holding": [], "waiting_for": None, "abortable": False}],
            extra_resources=(9,),
        )
        result = solve(payload)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["events"], [{"type": "complete", "job": 5, "released": []}])


# ---------------------------------------------------------------------------
# Deadlock resolution
# ---------------------------------------------------------------------------

class ResolutionTests(unittest.TestCase):
    def test_two_cycle_aborts_cheapest_job(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": True, "abort_cost": 5},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": True, "abort_cost": 3},
        ])
        result = solve(payload)
        self.assertEqual(result["status"], "resolved_with_aborts")
        self.assertEqual(result["aborted"], [2])
        self.assertEqual(result["abort_cost"], 3)
        self.assertEqual(result["events"], [
            {"type": "abort", "job": 2, "released": [2]},
            {"type": "grant", "job": 1, "resource": 2},
            {"type": "complete", "job": 1, "released": [1, 2]},
        ])

    def test_protected_holder_forces_abort_of_the_other_job(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": False},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": True, "abort_cost": 7},
        ])
        result = solve(payload)
        self.assertEqual(result["status"], "resolved_with_aborts")
        self.assertEqual(result["aborted"], [2])
        self.assertEqual(result["abort_cost"], 7)

    def test_three_cycle_tie_breaks_to_smallest_id(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": True, "abort_cost": 3},
            {"id": 2, "holding": [2], "waiting_for": 3, "abortable": True, "abort_cost": 3},
            {"id": 3, "holding": [3], "waiting_for": 1, "abortable": True, "abort_cost": 3},
        ])
        result = solve(payload)
        self.assertEqual(result["aborted"], [1])  # equal costs -> lexicographically smallest
        self.assertEqual(result["abort_cost"], 3)

    def test_lexicographic_tie_break_across_two_cycles(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": True, "abort_cost": 4},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": True, "abort_cost": 4},
            {"id": 3, "holding": [3], "waiting_for": 4, "abortable": True, "abort_cost": 1},
            {"id": 4, "holding": [4], "waiting_for": 3, "abortable": True, "abort_cost": 1},
        ])
        result = solve(payload)
        # candidates {1,3} {1,4} {2,3} {2,4} all cost 5 -> [1, 3] wins
        self.assertEqual(result["aborted"], [1, 3])
        self.assertEqual(result["abort_cost"], 5)

    def test_cheap_job_outside_the_cycle_is_not_aborted(self):
        # Job 3 is cheap but aborting it cannot break the 1<->2 cycle.
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": True, "abort_cost": 2},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": True, "abort_cost": 9},
            {"id": 3, "holding": [3], "waiting_for": 1, "abortable": True, "abort_cost": 1},
        ])
        result = solve(payload)
        self.assertEqual(result["aborted"], [1])
        self.assertEqual(result["abort_cost"], 2)
        # Job 3 must survive and complete once r1 is released.
        self.assertEqual(result["events"][-1], {"type": "complete", "job": 3, "released": [1, 3]})

    def test_abort_releases_all_held_resources(self):
        payload = mk_payload([
            {"id": 1, "holding": [1, 2], "waiting_for": 3, "abortable": True, "abort_cost": 1},
            {"id": 2, "holding": [3], "waiting_for": 1, "abortable": False},
        ])
        result = solve(payload)
        self.assertEqual(result["aborted"], [1])
        self.assertEqual(result["events"], [
            {"type": "abort", "job": 1, "released": [1, 2]},
            {"type": "grant", "job": 2, "resource": 1},
            {"type": "complete", "job": 2, "released": [1, 3]},
        ])

    def test_twelve_job_cycle_at_scale_limit(self):
        jobs = []
        n = MAX_JOBS
        for i in range(1, n + 1):
            jobs.append({
                "id": i,
                "holding": [i],
                "waiting_for": i % n + 1,
                "abortable": True,
                "abort_cost": (i % 3) + 1,
            })
        result = solve(mk_payload(jobs))
        self.assertEqual(result["status"], "resolved_with_aborts")
        # cheapest cost is 1, achieved by ids 3, 6, 9, 12 -> smallest id wins
        self.assertEqual(result["aborted"], [3])
        self.assertEqual(result["abort_cost"], 1)


# ---------------------------------------------------------------------------
# Unresolvable deadlocks (protected jobs)
# ---------------------------------------------------------------------------

class UnresolvableTests(unittest.TestCase):
    def test_protected_cycle_is_reported_not_force_released(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": False},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": False},
        ])
        snapshot = copy.deepcopy(payload)
        result = solve(payload)
        self.assertEqual(result["status"], "unresolvable")
        self.assertEqual(result["stuck"], [1, 2])
        self.assertEqual(result["protected_stuck"], [1, 2])
        self.assertIn("protected", result["message"].lower())
        self.assertEqual(result["events"], [])          # no progress, no aborts
        self.assertEqual(payload, snapshot)             # input untouched
        # Both protected jobs still hold their resources: nothing was released.
        active, _, holder = replay_events(payload, result["events"])
        self.assertEqual(active, {1, 2})
        self.assertEqual(holder[1], 1)
        self.assertEqual(holder[2], 2)

    def test_abortable_cycle_elsewhere_does_not_make_it_resolvable(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": False},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": False},
            {"id": 3, "holding": [3], "waiting_for": 4, "abortable": True, "abort_cost": 1},
            {"id": 4, "holding": [4], "waiting_for": 3, "abortable": True, "abort_cost": 1},
        ])
        result = solve(payload)
        self.assertEqual(result["status"], "unresolvable")
        self.assertEqual(result["stuck"], [1, 2, 3, 4])
        self.assertEqual(result["protected_stuck"], [1, 2])
        # No abort events at all: partial resolution is not performed.
        self.assertFalse(any(ev["type"] == "abort" for ev in result["events"]))

    def test_find_min_abort_set_returns_none_for_protected_cycle(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": False},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": False},
        ])
        state = build_state(payload)
        _, stuck = simulate(state)
        self.assertEqual(stuck, [1, 2])
        self.assertIsNone(find_min_abort_set(state, stuck))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def assert_invalid(self, payload, fragment):
        with self.subTest(fragment=fragment):
            with self.assertRaises(ValueError) as ctx:
                solve(payload)
            self.assertIn(fragment, str(ctx.exception))

    def test_validation_errors(self):
        base_job = {"id": 1, "holding": [], "waiting_for": None, "abortable": False}
        cases = [
            ({"jobs": [dict(base_job)] * 0 + [{"id": i, "holding": [], "waiting_for": None,
              "abortable": False} for i in range(MAX_JOBS + 1)], "resources": []},
             "at most 12 jobs"),
            ({"jobs": [], "resources": [{"id": i, "holder": None}
              for i in range(MAX_RESOURCES + 1)]},
             "at most 15 resources"),
            (mk_payload([dict(base_job), dict(base_job)]), "duplicate job id"),
            ({"jobs": [], "resources": [{"id": 1, "holder": None}, {"id": 1, "holder": None}]},
             "duplicate resource id"),
            ({"jobs": [dict(base_job)], "resources": [{"id": 1, "holder": 99}]},
             "held by unknown job"),
            ({"jobs": [dict(base_job, holding=[7])], "resources": []},
             "holds unknown resource"),
            ({"jobs": [dict(base_job, waiting_for=7)], "resources": []},
             "waits for unknown resource"),
            ({"jobs": [dict(base_job)], "resources": [{"id": 1, "holder": 1}]},
             "does not list it"),
            ({"jobs": [dict(base_job, holding=[1])], "resources": [{"id": 1, "holder": None}]},
             "record disagrees"),
            (mk_payload([{"id": 1, "holding": [1], "waiting_for": 1, "abortable": False}]),
             "already holds"),
            (mk_payload([dict(base_job, abortable=True, abort_cost=0)]),
             "positive integer"),
            (mk_payload([dict(base_job, abortable=True, abort_cost=-3)]),
             "positive integer"),
            (mk_payload([dict(base_job, abortable=True)]),
             "positive integer"),
            (mk_payload([dict(base_job, abortable=True, abort_cost=True)]),
             "positive integer"),
            (mk_payload([dict(base_job, abort_cost=5)]),
             "must not carry"),
            ({"jobs": "nope", "resources": []}, "must be lists"),
            ("nope", "payload must be an object"),
            (mk_payload([dict(base_job, id="a")]), "job id must be an integer"),
        ]
        for payload, fragment in cases:
            self.assert_invalid(payload, fragment)

    def test_scale_limits_are_accepted(self):
        jobs = [{"id": i, "holding": [i], "waiting_for": None, "abortable": False}
                for i in range(MAX_JOBS)]
        payload = mk_payload(jobs, extra_resources=range(MAX_JOBS, MAX_RESOURCES))
        self.assertEqual(len(payload["resources"]), MAX_RESOURCES)
        self.assertEqual(solve(payload)["status"], "completed")


# ---------------------------------------------------------------------------
# Randomised cross-check: independent replay + subset enumeration
# ---------------------------------------------------------------------------

class RandomCrossCheckTests(unittest.TestCase):
    def test_random_instances(self):
        for seed in range(150):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                payload = random_payload(rng)
                snapshot = copy.deepcopy(payload)
                result = solve(payload)

                self.assertEqual(payload, snapshot, "solve must not mutate its input")
                json.dumps(result)  # the reply must be JSON-serialisable

                # Independent replay of the returned event stream.
                active, _, holder = replay_events(payload, result["events"])

                # Independent minimality check by subset enumeration.
                expected = brute_force_min_abort_set(payload)

                if result["status"] == "unresolvable":
                    self.assertIsNone(expected)
                    self.assertEqual(sorted(active), result["stuck"])
                    for jid in result["protected_stuck"]:
                        self.assertIn(jid, active)
                    # Nothing was force-released: stuck jobs keep resources.
                    for r in payload["resources"]:
                        if r["holder"] in result["stuck"]:
                            self.assertEqual(holder[r["id"]], r["holder"])
                else:
                    self.assertEqual(active, set(), "all jobs must complete")
                    self.assertEqual(result["aborted"], expected or [])
                    costs = {j["id"]: j.get("abort_cost") for j in payload["jobs"]}
                    if result["status"] == "resolved_with_aborts":
                        self.assertEqual(
                            result["abort_cost"],
                            sum(costs[j] for j in result["aborted"]))
                    else:
                        self.assertEqual(result["status"], "completed")
                        self.assertEqual(expected, [])


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------

class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()

    def _post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    def test_simulate_endpoint(self):
        payload = mk_payload([
            {"id": 1, "holding": [1], "waiting_for": 2, "abortable": True, "abort_cost": 5},
            {"id": 2, "holding": [2], "waiting_for": 1, "abortable": True, "abort_cost": 3},
        ])
        status, data = self._post("/simulate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "resolved_with_aborts")
        self.assertEqual(data["aborted"], [2])

    def test_invalid_payload_returns_400(self):
        status, data = self._post("/simulate", {"jobs": [{"id": "x"}], "resources": []})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_health_endpoint(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("GET", "/health")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read()), {"status": "ok"})
        conn.close()


if __name__ == "__main__":
    unittest.main()
