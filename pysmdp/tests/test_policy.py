import math
import random

import pytest
import torch

from smdp.grid import N_ACTIONS, Grid, make_state
from smdp.mdp import Params
from smdp.policy import Policy
from smdp.solver import Transitions, value_iterate_l0

PARAMS = Params()
GOAL = (0, 1)
STATE = make_state((4, 0), (4, 4), (2, 3), (2, 1))


@pytest.fixture(scope="module")
def solved():
    T = Transitions(Grid(n_side=5))
    return T, value_iterate_l0(T, 0, GOAL, PARAMS)


def test_q_matches_values(solved):
    T, solution = solved
    assert torch.allclose(solution.q.max(dim=0).values[T.valid], solution.values[T.valid], atol=1e-3)


def test_probabilities_sum_to_one(solved):
    T, solution = solved
    policy = Policy.from_q(0, solution.q, PARAMS.tau)
    sums = policy.probs[:, T.valid].sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_matches_softmax_formula(solved):
    T, solution = solved
    policy = Policy.from_q(0, solution.q, PARAMS.tau)
    q = solution.q[:, T.grid.encode(STATE)].tolist()
    weights = [math.exp(PARAMS.tau * v) for v in q]
    expected = [w / sum(weights) for w in weights]
    assert policy.probs_at(T.grid, STATE) == pytest.approx(expected, abs=1e-4)


def test_zero_temperature_is_uniform(solved):
    T, solution = solved
    policy = Policy.from_q(0, solution.q, tau=0)
    assert policy.probs_at(T.grid, STATE) == pytest.approx([1 / N_ACTIONS] * N_ACTIONS)


def test_higher_temperature_is_greedier(solved):
    T, solution = solved
    q = solution.q[:, T.grid.encode(STATE)].tolist()
    best = max(range(N_ACTIONS), key=q.__getitem__)
    best_probs = [Policy.from_q(0, solution.q, tau).prob(T.grid, STATE, best) for tau in (0, 1, 5, 50)]
    assert best_probs == sorted(best_probs)
    assert best_probs[-1] > 0.45  # at most two actions tie for best in this state


def test_sampling_follows_probabilities(solved):
    T, solution = solved
    policy = Policy.from_q(0, solution.q, PARAMS.tau)
    rng = random.Random(0)
    n = 20000
    counts = [0] * N_ACTIONS
    for _ in range(n):
        counts[policy.sample(T.grid, STATE, rng)] += 1
    assert [c / n for c in counts] == pytest.approx(policy.probs_at(T.grid, STATE), abs=0.015)
