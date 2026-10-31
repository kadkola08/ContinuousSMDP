"""Running agents in the grid world."""

import random
from collections.abc import Callable

from smdp.grid import Grid, State
from smdp.render import is_blocked, step_title

# choose(t, s, p) -> action for agent p at time step t in state s.
Chooser = Callable[[int, State, int], int]


def greedy_chooser(grid: Grid, q_tables, seed: int = 0) -> Chooser:
    """Agent p picks the action with the highest Q in q_tables[p], (n_actions, n_states).

    Ties are broken at random, reproducibly for a given seed.
    """
    rng = random.Random(seed)

    def choose(t, s, p):
        q = q_tables[p][:, grid.encode(s)].tolist()
        best = max(q)
        return rng.choice([a for a, v in enumerate(q) if best - v < 1e-6])

    return choose


def rollout(grid: Grid, start: State, choose: Chooser, steps: int, halves: list | None = None):
    """Runs ``steps`` joint moves; agent 0 moves first and agent 1 sees the result.

    Returns the visited states and a title per frame. If ``halves`` is given, the state after
    agent 0's move in each step is appended to it, so agent 1's move t goes from halves[t - 1]
    to states[t].
    """
    states, titles = [start], [step_title(0, None, (False, False))]
    s = start
    for t in range(1, steps + 1):
        a = choose(t, s, 0)
        s1 = grid.step(s, 0, a)
        b = choose(t, s1, 1)
        s2 = grid.step(s1, 1, b)
        titles.append(step_title(t, (a, b), (is_blocked(s, s1, a), is_blocked(s1, s2, b))))
        states.append(s2)
        if halves is not None:
            halves.append(s1)
        s = s2
    return states, titles
