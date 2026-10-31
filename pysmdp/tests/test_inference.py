import pytest
import torch

from smdp.grid import DOWN, N_ACTIONS, RIGHT, STAY, Grid, make_state
from smdp.inference import GoalEstimator, transition_likelihood
from smdp.mdp import Params
from smdp.policy import Policy
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, level0_policies, value_iterate_l0

PARAMS = Params()
FLOWER, TREE = (0, 1), (0, 3)
TINY = Grid(n_side=3)


def uniform(agent, grid):
    return Policy(agent, torch.full((N_ACTIONS, grid.n_states), 1 / N_ACTIONS))


def always(agent, grid, action):
    probs = torch.zeros((N_ACTIONS, grid.n_states))
    probs[action] = 1.0
    return Policy(agent, probs)


def test_blocked_moves_count_as_staying():
    # Agent 0 in the top-left corner: staying, up and left all leave it in place.
    s = make_state((0, 0), (2, 2), (2, 0), (0, 2))
    assert transition_likelihood(TINY, uniform(0, TINY), s, s) == pytest.approx(3 / N_ACTIONS)
    moved = TINY.step(s, 0, RIGHT)
    assert transition_likelihood(TINY, uniform(0, TINY), s, moved) == pytest.approx(1 / N_ACTIONS)


def test_bayes_update():
    s = make_state((1, 1), (2, 2), (2, 0), (0, 2))
    moved = TINY.step(s, 0, DOWN)
    policies = {"a": always(0, TINY, DOWN), "b": uniform(0, TINY)}
    estimator = GoalEstimator(TINY, policies, prior={"a": 0.2, "b": 0.8})
    estimator.update(s, moved)
    # Likelihoods 1 and 1/5: posterior ∝ (0.2, 0.16).
    assert estimator.probs["a"] == pytest.approx(0.2 / 0.36)
    assert estimator.probs["b"] == pytest.approx(0.16 / 0.36)


def test_impossible_move_keeps_belief():
    s = make_state((1, 1), (2, 2), (2, 0), (0, 2))
    estimator = GoalEstimator(TINY, {"a": always(0, TINY, STAY), "b": always(0, TINY, STAY)})
    estimator.update(s, TINY.step(s, 0, DOWN))
    assert estimator.probs == {"a": 0.5, "b": 0.5}


@pytest.fixture(scope="module")
def setup():
    T = Transitions(Grid(n_side=5))
    return T, level0_policies(T, 0, [FLOWER, TREE], PARAMS)


@pytest.mark.parametrize("true_goal", [FLOWER, TREE])
def test_belief_converges_to_true_goal(setup, true_goal):
    T, policies = setup
    yellow = value_iterate_l0(T, 0, true_goal, PARAMS)
    stay = torch.zeros_like(yellow.q)
    stay[STAY] = 1.0  # red stays out of the way
    start = make_state((4, 0), (4, 4), (2, 3), (2, 1))
    # For both goals yellow first pushes its can straight up, so the first ~6 moves carry
    # almost no evidence; 15 steps leaves time for the belief to settle.
    states, _ = rollout(T.grid, start, greedy_chooser(T.grid, [yellow.q, stay]), 15)

    estimator = GoalEstimator(T.grid, policies)
    for before, after in zip(states, states[1:]):
        estimator.update(before, after)
    assert estimator.probs[true_goal] > 0.9
    assert sum(estimator.probs.values()) == pytest.approx(1.0)
