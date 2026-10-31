import pytest
import torch

from smdp.grid import N_ACTIONS, STAY, Grid, make_state, other_agent
from smdp.mdp import Params, reward, reward_batch
from smdp.solver import Transitions, q_at, value_iterate_l0

PARAMS = Params()


def test_reward_peaks_at_goal_and_fades_with_distance():
    goal = (0, 0)
    on_goal = make_state((4, 4), (4, 3), (0, 0), (3, 3))
    near = make_state((4, 4), (4, 3), (0, 1), (3, 3))
    far = make_state((4, 4), (4, 3), (4, 4 - 1), (3, 3))
    assert reward(on_goal, 0, goal, PARAMS) == PARAMS.rho
    assert PARAMS.rho > reward(near, 0, goal, PARAMS) > 0
    assert reward(far, 0, goal, PARAMS) == 0  # distance 5 = delta


def test_reward_uses_own_can():
    s = make_state((4, 4), (4, 3), (0, 0), (3, 3))
    assert reward(s, 0, (0, 0), PARAMS) == PARAMS.rho
    assert reward(s, 1, (0, 0), PARAMS) < PARAMS.rho


def test_reward_batch_matches_scalar():
    grid = Grid(n_side=3)
    states = grid.decode_range(0, grid.n_states)
    for p in (0, 1):
        batch = reward_batch(states, p, (1, 2), PARAMS)
        for i in [0, 100, 2000, grid.n_states - 1]:
            assert batch[i].item() == pytest.approx(reward(grid.decode(i), p, (1, 2), PARAMS))


def reference_value_iteration(grid, p, goal, params, sweeps):
    """Plain loop over states with the scalar step function."""
    states = [grid.decode(i) for i in range(grid.n_states)]
    values = [0.0] * grid.n_states
    other = other_agent(p)
    for _ in range(sweeps):
        new = [0.0] * grid.n_states
        for i, s in enumerate(states):
            if not grid.is_valid(s):
                continue
            q = []
            for a in range(N_ACTIONS):
                s1 = grid.step(s, p, a)
                future = sum(values[grid.encode(grid.step(s1, other, b))] for b in range(N_ACTIONS))
                cost = params.action_cost(a)
                q.append(reward(s, p, goal, params) - cost + params.gamma * future / N_ACTIONS)
            new[i] = max(q)
        values = new
    return torch.tensor(values)


@pytest.mark.parametrize("p", [0, 1])
def test_matches_reference_on_tiny_grid(p):
    grid = Grid(n_side=3)
    T = Transitions(grid)
    sweeps = 6
    solution = value_iterate_l0(T, p, (0, 2), PARAMS, tol=0, max_iters=sweeps)
    expected = reference_value_iteration(grid, p, (0, 2), PARAMS, sweeps)
    assert torch.allclose(solution.values, expected, atol=1e-5)


@pytest.fixture(scope="module")
def solved():
    grid = Grid(n_side=5)
    T = Transitions(grid)
    goal = (0, 1)
    return T, goal, value_iterate_l0(T, 0, goal, PARAMS)


def test_converges(solved):
    _, _, solution = solved
    assert solution.residual < 1e-4


def test_greedy_policy_brings_can_to_goal(solved):
    T, goal, solution = solved
    grid = T.grid
    s = make_state((4, 0), (4, 4), (2, 3), (2, 1))
    for _ in range(20):
        q = solution.q[:, grid.encode(s)].tolist()
        s = grid.step(s, 0, max(range(N_ACTIONS), key=q.__getitem__))
        s = grid.step(s, 1, STAY)  # the other agent stays out of the way
        if (s[4], s[5]) == goal:
            break
    assert (s[4], s[5]) == goal


def test_staying_is_best_with_can_on_goal(solved):
    T, goal, solution = solved
    s = make_state((1, 2), (4, 4), goal, (2, 3))
    q = solution.q[:, T.grid.encode(s)].tolist()
    assert max(range(N_ACTIONS), key=q.__getitem__) == STAY
