import pytest
import torch

from smdp.grid import DOWN, N_ACTIONS, STAY, Grid, make_state
from smdp.inference import GoalEstimator
from smdp.mdp import Params
from smdp.policy import Policy
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, level0_policy_mixture, level1_policies, value_iterate_l0

PARAMS = Params()
FLOWER, TREE = (0, 1), (0, 3)
# Red's can is on the tree and red stands below the yellow can (run_l1.py's "blocking").
BLOCKING = make_state((4, 4), (3, 1), (2, 1), TREE)
TINY = Grid(n_side=3)


def always(agent, action):
    probs = torch.zeros((N_ACTIONS, TINY.n_states))
    probs[action] = 1.0
    return Policy(agent, probs)


def test_marginals_of_joint_belief():
    policies = {(g, chi): always(1, STAY) for g in "ab" for chi in (-1, 1)}
    estimator = GoalEstimator(
        TINY, policies, prior={("a", -1): 0.1, ("a", 1): 0.2, ("b", -1): 0.3, ("b", 1): 0.4}
    )
    assert estimator.marginal(0) == pytest.approx({"a": 0.3, "b": 0.7})
    assert estimator.marginal(1) == pytest.approx({-1: 0.4, 1: 0.6})


def test_policies_can_change_between_updates():
    s = make_state((2, 2), (1, 1), (2, 0), (0, 2))
    moved = TINY.step(s, 1, DOWN)
    estimator = GoalEstimator(TINY, {"h1": always(1, STAY), "h2": always(1, STAY)})
    estimator.update(s, moved, policies={"h1": always(1, DOWN), "h2": always(1, STAY)})
    assert estimator.probs == {"h1": 1.0, "h2": 0.0}


@pytest.fixture(scope="module")
def setup():
    """Red's level-1 policies for chi in {-1, 0, 1}; red's goal is the tree and red knows
    yellow wants the flower. Yellow acts as a level-0 agent."""
    T = Transitions(Grid(n_side=5))
    belief_about_yellow = [(FLOWER, 1.0)]
    yellow_model = level0_policy_mixture(T, 0, belief_about_yellow, PARAMS)
    hypotheses = level1_policies(
        T, 1, [TREE], [-1.0, 0.0, 1.0], belief_about_yellow, yellow_model, PARAMS
    )
    yellow = value_iterate_l0(T, 0, FLOWER, PARAMS)
    return T, hypotheses, yellow


def infer_chi(T, hypotheses, yellow, true_chi, steps=8):
    """Red acts greedily with true_chi; yellow watches red's moves. Returns P(chi)."""
    red_q = torch.log(hypotheses[(TREE, true_chi)].probs)  # same argmax as red's Q
    halves = []
    states, _ = rollout(T.grid, BLOCKING, greedy_chooser(T.grid, [yellow.q, red_q]), steps, halves)
    estimator = GoalEstimator(T.grid, hypotheses)
    for before, after in zip(halves, states[1:]):  # red moves from halves[t] to states[t + 1]
        estimator.update(before, after)
    return estimator.marginal(1)


def test_helper_is_recognized(setup):
    belief = infer_chi(*setup, true_chi=1.0)
    assert max(belief, key=belief.get) == 1.0


def test_hinderer_is_recognized(setup):
    belief = infer_chi(*setup, true_chi=-1.0)
    assert max(belief, key=belief.get) == -1.0
