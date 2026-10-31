import math

import pytest
import torch

from smdp.grid import N_ACTIONS, Grid, make_state
from smdp.mdp import Params
from smdp.policy import Policy
from smdp.solver import (
    Transitions,
    level0_policy_mixture,
    q_at,
    social_rewards,
    value_iterate,
    value_iterate_l0,
    value_iterate_l1,
)

PARAMS = Params()
FLOWER, TREE = (0, 1), (0, 3)


def test_mixture_weights_policies():
    a = Policy(0, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    b = Policy(0, torch.tensor([[0.0, 0.5], [1.0, 0.5]]))
    mixed = Policy.mixture([(0.25, a), (0.75, b)])
    assert torch.allclose(mixed.probs, torch.tensor([[0.25, 0.375], [0.75, 0.625]]))


def test_chi_zero_with_uniform_other_equals_level0():
    T = Transitions(Grid(n_side=3))
    uniform = Policy(1, torch.full((N_ACTIONS, T.grid.n_states), 1 / N_ACTIONS))
    l0 = value_iterate_l0(T, 0, (0, 2), PARAMS)
    l1 = value_iterate_l1(T, 0, (0, 2), 0.0, [((2, 0), 1.0)], uniform, PARAMS)
    assert torch.allclose(l0.values, l1.values, atol=1e-5)


@pytest.mark.parametrize("chi", [-1.0, 1.0])
def test_batched_q_matches_scalar_reference(chi):
    T = Transitions(Grid(n_side=3))
    belief = [((2, 0), 0.7), ((0, 0), 0.3)]
    other = level0_policy_mixture(T, 1, belief, PARAMS)
    solution = value_iterate_l1(T, 0, (0, 2), chi, belief, other, PARAMS)
    rewards = social_rewards(T, 0, (0, 2), chi, belief, PARAMS)
    for index in [5, 300, 2000, 4000, 6000]:
        s = T.grid.decode(index)
        if not T.grid.is_valid(s):
            continue
        expected = q_at(T, solution.values, s, 0, rewards, PARAMS, other.probs)
        assert solution.q[:, index].tolist() == pytest.approx(expected, abs=1e-4)


@pytest.fixture(scope="module")
def red_l1():
    """Red's level-1 Q tables for several chi; red knows yellow wants the flower."""
    T = Transitions(Grid(n_side=5))
    belief = [(FLOWER, 1.0)]
    yellow = level0_policy_mixture(T, 0, belief, PARAMS)
    return T, {
        chi: value_iterate_l1(T, 1, TREE, chi, belief, yellow, PARAMS).q for chi in (-1.0, 0.0, 1.0)
    }


def yellow_can_distance_after_red_acts(T, q, steps=12):
    """Red's can is already on the tree; yellow stays in a corner while red acts greedily."""
    s = make_state((4, 0), (3, 4), (2, 2), TREE)
    for _ in range(steps):
        column = q[:, T.grid.encode(s)].tolist()
        s = T.grid.step(s, 1, max(range(N_ACTIONS), key=column.__getitem__))
    return math.dist((s[4], s[5]), FLOWER), s


def test_helping_brings_other_can_closer(red_l1):
    T, q = red_l1
    start = math.dist((2, 2), FLOWER)
    distance, _ = yellow_can_distance_after_red_acts(T, q[1.0])
    assert distance < start


def test_hindering_pushes_other_can_away(red_l1):
    T, q = red_l1
    start = math.dist((2, 2), FLOWER)
    distance, _ = yellow_can_distance_after_red_acts(T, q[-1.0])
    assert distance > start


def test_no_social_goal_leaves_other_can_alone(red_l1):
    T, q = red_l1
    distance, s = yellow_can_distance_after_red_acts(T, q[0.0])
    assert distance == math.dist((2, 2), FLOWER)
    assert (s[6], s[7]) == TREE
