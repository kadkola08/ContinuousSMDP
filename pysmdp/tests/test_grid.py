import pytest
import torch

from smdp.grid import ACTIONS, DOWN, LEFT, N_ACTIONS, RIGHT, STAY, UP, Grid, make_state

GRID = Grid(n_side=5)


def test_encode_decode_roundtrip():
    for index in [0, 1, 12345, GRID.n_states - 1]:
        assert GRID.encode(GRID.decode(index)) == index


def test_is_valid():
    assert Grid.is_valid(make_state((0, 0), (0, 1), (1, 0), (1, 1)))
    assert not Grid.is_valid(make_state((0, 0), (0, 0), (1, 0), (1, 1)))
    assert not Grid.is_valid(make_state((0, 0), (0, 1), (2, 2), (2, 2)))


def test_stay():
    s = make_state((2, 2), (0, 0), (4, 4), (4, 0))
    assert GRID.step(s, 0, STAY) == s


@pytest.mark.parametrize("action", [RIGHT, LEFT, DOWN, UP])
def test_walk(action):
    s = make_state((2, 2), (0, 0), (4, 4), (4, 0))
    dr, dc = ACTIONS[action]
    assert GRID.step(s, 0, action) == make_state((2 + dr, 2 + dc), (0, 0), (4, 4), (4, 0))


def test_wall_blocks():
    s = make_state((0, 4), (2, 2), (4, 4), (4, 0))
    assert GRID.step(s, 0, UP) == s
    assert GRID.step(s, 0, RIGHT) == s


def test_other_agent_blocks():
    s = make_state((2, 2), (2, 3), (4, 4), (4, 0))
    assert GRID.step(s, 0, RIGHT) == s
    assert GRID.step(s, 1, LEFT) == s


def test_push_own_can():
    s = make_state((2, 1), (0, 0), (2, 2), (4, 0))
    assert GRID.step(s, 0, RIGHT) == make_state((2, 2), (0, 0), (2, 3), (4, 0))


def test_push_other_agents_can():
    s = make_state((2, 1), (0, 0), (4, 4), (2, 2))
    assert GRID.step(s, 0, RIGHT) == make_state((2, 2), (0, 0), (4, 4), (2, 3))


def test_push_into_wall_moves_nothing():
    s = make_state((2, 3), (0, 0), (2, 4), (4, 0))
    assert GRID.step(s, 0, RIGHT) == s


def test_push_into_other_agent_moves_nothing():
    s = make_state((2, 1), (2, 3), (2, 2), (4, 0))
    assert GRID.step(s, 0, RIGHT) == s


def test_chain_push():
    s = make_state((2, 0), (0, 0), (2, 1), (2, 2))
    assert GRID.step(s, 0, RIGHT) == make_state((2, 1), (0, 0), (2, 2), (2, 3))
    # Same with the cans in the other order.
    s = make_state((2, 0), (0, 0), (2, 2), (2, 1))
    assert GRID.step(s, 0, RIGHT) == make_state((2, 1), (0, 0), (2, 3), (2, 2))


def test_chain_push_into_wall_moves_nothing():
    s = make_state((2, 2), (0, 0), (2, 3), (2, 4))
    assert GRID.step(s, 0, RIGHT) == s


def test_chain_push_into_other_agent_moves_nothing():
    s = make_state((2, 0), (2, 3), (2, 1), (2, 2))
    assert GRID.step(s, 0, RIGHT) == s


def test_batch_decode_encode_match_scalar():
    grid = Grid(n_side=3)
    states = grid.decode_range(0, grid.n_states)
    assert torch.equal(grid.encode_batch(states), torch.arange(grid.n_states))
    for index in [0, 7, 1000, grid.n_states - 1]:
        assert tuple(states[index].tolist()) == grid.decode(index)


@pytest.mark.parametrize("n_side", [3, 4])
def test_batch_matches_scalar_on_all_valid_states(n_side):
    grid = Grid(n_side=n_side)
    states = grid.decode_range(0, grid.n_states)
    valid = grid.is_valid_batch(states)
    expected_valid = torch.tensor([Grid.is_valid(grid.decode(i)) for i in range(grid.n_states)])
    assert torch.equal(valid, expected_valid)

    states = states[valid]
    scalar_states = [tuple(s) for s in states.tolist()]
    for p in (0, 1):
        for action in range(N_ACTIONS):
            batch_next = grid.step_batch(states, p, action).tolist()
            for s, got in zip(scalar_states, batch_next):
                assert tuple(got) == grid.step(s, p, action), (s, p, action)


def test_step_keeps_states_valid():
    grid = Grid(n_side=4)
    states = grid.decode_range(0, grid.n_states)
    states = states[grid.is_valid_batch(states)]
    for p in (0, 1):
        for action in range(N_ACTIONS):
            assert grid.is_valid_batch(grid.step_batch(states, p, action)).all()
