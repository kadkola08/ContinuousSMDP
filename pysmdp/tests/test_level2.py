import pytest
import torch

from smdp.grid import Grid, make_state
from smdp.mdp import Params, reward
from smdp.policy import Policy
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import (
    Transitions,
    level0_policy_mixture,
    level1_policies,
    level2_rewards,
    q_at,
    value_iterate_l0,
    value_iterate_l2,
)

PARAMS = Params()
FLOWER, TREE = (0, 1), (0, 3)
BLOCKING = make_state((4, 4), (3, 1), (2, 1), TREE)  # run_l1.py's "blocking" scenario


def test_level2_rewards_match_formula():
    T = Transitions(Grid(n_side=3))
    other_belief = [(((2, 0), 1.0), 0.6), (((0, 0), -0.5), 0.4)]
    about_me = [((0, 2), 0.7), ((2, 2), 0.3)]
    chi = 0.5
    rewards = level2_rewards(T, 0, (0, 2), chi, other_belief, about_me, PARAMS)
    for index in [10, 500, 3000, 6000]:
        s = T.grid.decode(index)
        mine_seen = sum(p * reward(s, 0, g, PARAMS) for g, p in about_me)
        other = sum(p * (reward(s, 1, g, PARAMS) + c * mine_seen) for (g, c), p in other_belief)
        expected = reward(s, 0, (0, 2), PARAMS) + chi * other
        assert rewards[index].item() == pytest.approx(expected, abs=1e-5)


def test_batched_q_matches_scalar_reference():
    T = Transitions(Grid(n_side=3))
    about_me = [((0, 2), 1.0)]
    me_model = level0_policy_mixture(T, 0, about_me, PARAMS)
    other = level1_policies(T, 1, [(2, 0)], [1.0, -1.0], about_me, me_model, PARAMS)
    other_belief = [(((2, 0), 1.0), 0.5), (((2, 0), -1.0), 0.5)]
    other_policy = Policy.mixture([(p, other[h]) for h, p in other_belief])
    solution = value_iterate_l2(T, 0, (0, 2), 0.5, other_belief, about_me, other_policy, PARAMS)
    rewards = level2_rewards(T, 0, (0, 2), 0.5, other_belief, about_me, PARAMS)
    for index in [5, 300, 2000, 4000, 6000]:
        s = T.grid.decode(index)
        if T.grid.is_valid(s):
            expected = q_at(T, solution.values, s, 0, rewards, PARAMS, other_policy.probs)
            assert solution.q[:, index].tolist() == pytest.approx(expected, abs=1e-4)


@pytest.fixture(scope="module")
def blocking():
    """Red is level 1 (goal tree, knows yellow wants the flower). Yellow's level-0 solution,
    and a level-2 yellow (chi = 0) told red's true chi, for a helping and a hindering red."""
    T = Transitions(Grid(n_side=5))
    about_yellow = [(FLOWER, 1.0)]
    yellow_model = level0_policy_mixture(T, 0, about_yellow, PARAMS)
    red = level1_policies(T, 1, [TREE], [1.0, -1.0], about_yellow, yellow_model, PARAMS)
    y0 = value_iterate_l0(T, 0, FLOWER, PARAMS)
    y2 = {
        chi: value_iterate_l2(T, 0, FLOWER, 0.0, [((TREE, chi), 1.0)], about_yellow, red[(TREE, chi)], PARAMS)
        for chi in (1.0, -1.0)
    }
    return T, red, y0, y2


def play(T, yellow_q, red_policy, steps=12):
    red_q = torch.log(red_policy.probs)  # same argmax as red's Q
    states, titles = rollout(T.grid, BLOCKING, greedy_chooser(T.grid, [yellow_q, red_q]), steps)
    moves = sum("Y: stay" not in title for title in titles[1:])
    total = sum(reward(s, 0, FLOWER, PARAMS) for s in states)
    return moves, total


def test_level2_lets_a_helper_help(blocking):
    T, red, y0, y2 = blocking
    moves0, total0 = play(T, y0.q, red[(TREE, 1.0)])
    moves2, total2 = play(T, y2[1.0].q, red[(TREE, 1.0)])
    assert total2 == pytest.approx(total0)  # red delivers the can either way
    assert moves2 < moves0  # but the level-2 yellow doesn't waste moves


def test_level2_does_better_against_a_hinderer(blocking):
    T, red, y0, y2 = blocking
    _, total0 = play(T, y0.q, red[(TREE, -1.0)])
    _, total2 = play(T, y2[-1.0].q, red[(TREE, -1.0)])
    assert total2 > total0
