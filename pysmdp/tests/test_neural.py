import pytest
import torch

from smdp.grid import N_ACTIONS, Grid
from smdp.mdp import Params, reward_batch
from smdp.neural import (
    Hypothesis,
    NeuralQ,
    bellman_targets,
    egocentric,
    level_rewards,
    other_policy,
    q_table,
    reward_rows,
    sample_simplex,
    sample_states,
)
from smdp.solver import (
    Transitions,
    level0_policy_mixture,
    level2_rewards,
    q_values,
    social_rewards,
    value_iterate_l0,
    value_iterate_l1,
)

PARAMS = Params()
LANDMARKS = [(0, 2), (2, 0)]
CHIS = [-1.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def T():
    return Transitions(Grid(n_side=3))


def valid_states(T):
    return T.states[T.valid]


def rows(n, *values):
    return torch.tensor([values] * n, dtype=torch.float32)


def test_swapping_agents_is_a_symmetry(T):
    s = valid_states(T)
    for a in range(N_ACTIONS):
        assert torch.equal(T.grid.step_batch(s, 1, a), egocentric(T.grid.step_batch(egocentric(s, 1), 0, a), 1))
    assert torch.equal(reward_rows(s, 1, torch.tensor([0.0, 2.0]), PARAMS),
                       reward_rows(egocentric(s, 1), 0, torch.tensor([0.0, 2.0]), PARAMS))


def test_reward_rows_matches_reward_batch(T):
    s = valid_states(T)
    for p in (0, 1):
        assert torch.allclose(reward_rows(s, p, torch.tensor([2.0, 0.0]), PARAMS), reward_batch(s, p, (2, 0), PARAMS))


def test_level1_rewards_match_solver(T):
    model = NeuralQ(1, 3, LANDMARKS, CHIS, PARAMS, width=8, depth=1)
    s = valid_states(T)
    h = Hypothesis(goal=rows(len(s), 0, 2), chi=torch.full((len(s),), -0.5), belief=rows(len(s), 0.3, 0.7))
    expected = social_rewards(T, 0, (0, 2), -0.5, [((0, 2), 0.3), ((2, 0), 0.7)], PARAMS)[T.valid]
    assert torch.allclose(level_rewards(model, s, h), expected, atol=1e-6)


def test_level2_rewards_match_solver(T):
    model = NeuralQ(2, 3, LANDMARKS, CHIS, PARAMS, width=8, depth=1)
    s = valid_states(T)
    joint = [0.1, 0.0, 0.2, 0.0, 0.4, 0.3]  # (landmark, chi) pairs in row-major order
    h = Hypothesis(goal=rows(len(s), 2, 0), chi=torch.full((len(s),), 1.5),
                   joint=rows(len(s), *joint), about_me=rows(len(s), 0.6, 0.4))
    other_belief = [((g, c), joint[k * len(CHIS) + m]) for k, g in enumerate(LANDMARKS) for m, c in enumerate(CHIS)]
    expected = level2_rewards(T, 0, (2, 0), 1.5, other_belief, [((0, 2), 0.6), ((2, 0), 0.4)], PARAMS)[T.valid]
    assert torch.allclose(level_rewards(model, s, h), expected, atol=1e-6)


@pytest.mark.parametrize("level", [0, 1])
def test_bellman_targets_match_tabular_backup(T, level):
    """With the tabular values plugged in, the sampled backup is exactly smdp.solver's."""
    goal, belief = (0, 2), [((0, 2), 0.25), ((2, 0), 0.75)]
    if level == 0:
        other = None
        solution = value_iterate_l0(T, 0, goal, PARAMS)
        rewards = reward_batch(T.states, 0, goal, PARAMS)
    else:
        other = level0_policy_mixture(T, 1, belief, PARAMS)
        solution = value_iterate_l1(T, 0, goal, 1.0, belief, other, PARAMS)
        rewards = social_rewards(T, 0, goal, 1.0, belief, PARAMS)
    rewards[~T.valid] = 0
    expected = q_values(solution.values, rewards, T, 0, PARAMS, None if other is None else other.probs)

    s = valid_states(T)
    index = T.grid.encode_batch
    uniform = lambda x, h: torch.full((len(x), N_ACTIONS), 1 / N_ACTIONS)
    targets = bellman_targets(
        T.grid, s, Hypothesis(goal=rows(len(s), *goal)), rewards[index(s)], PARAMS,
        value_fn=lambda x, h: solution.values[index(x)],
        other_fn=uniform if other is None else (lambda x, h: other.probs[:, index(x)].T),
    )
    assert torch.allclose(targets, expected[:, T.valid].T, atol=1e-5)


def test_other_policy_mixes_lower_level_in_the_other_agents_frame(T):
    """A stand-in level-0 'network' that looks up agent 1's tabular Q gives the solver's mixture."""
    tables = {g: value_iterate_l0(T, 1, g, PARAMS).q for g in LANDMARKS}

    def lower(view, h):  # view is agent 1's frame; swapping back gives world states
        goal = tuple(int(c) for c in h.goal[0].tolist())
        return tables[goal][:, T.grid.encode_batch(egocentric(view, 1))].T

    model = NeuralQ(1, 3, LANDMARKS, CHIS, PARAMS, width=8, depth=1)
    s = valid_states(T)
    h = Hypothesis(goal=rows(len(s), 0, 2), chi=torch.zeros(len(s)), belief=rows(len(s), 0.4, 0.6))
    expected = level0_policy_mixture(T, 1, [((0, 2), 0.4), ((2, 0), 0.6)], PARAMS).probs[:, T.valid].T
    assert torch.allclose(other_policy(model, lower, s, h), expected, atol=1e-6)


def test_samplers():
    grid = Grid(n_side=3)
    s = sample_states(grid, 500)
    assert len(s) == 500 and grid.is_valid_batch(s).all()
    b = sample_simplex(1000, 6)
    assert (b >= 0).all() and torch.allclose(b.sum(dim=1), torch.ones(1000))


def test_q_table_matches_direct_evaluation(T):
    model = NeuralQ(1, 3, LANDMARKS, CHIS, PARAMS, width=16, depth=2)
    table = q_table(model, T.grid, 1, chunk=1000, goal=(2, 0), chi=0.5, belief=[0.2, 0.8])
    s = valid_states(T)
    with torch.no_grad():
        direct = model.q(s, 1, Hypothesis.constant(len(s), goal=(2, 0), chi=0.5, belief=[0.2, 0.8]))
    assert torch.allclose(table[:, T.valid].T, direct, atol=1e-6)
    assert (table[:, ~T.valid] == 0).all()


def test_save_and_load(tmp_path):
    model = NeuralQ(2, 3, LANDMARKS, CHIS, PARAMS, width=16, depth=2)
    model.save(tmp_path / "q2.pt")
    loaded = NeuralQ.load(tmp_path / "q2.pt")
    s = sample_states(Grid(n_side=3), 10)
    h = Hypothesis(goal=rows(10, 0, 2), chi=torch.ones(10), joint=sample_simplex(10, 6), about_me=sample_simplex(10, 2))
    assert torch.equal(model(s, h), loaded(s, h))
