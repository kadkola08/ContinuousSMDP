"""Solves the level-0 MDP for both agents and visualizes them acting greedily.

    python run_l0.py
    python run_l0.py --neural out/neural    # the level-0 network instead of value iteration

Yellow wants its can on the flower and red wants its can on the tree. Each agent plans as if
the other one moves randomly. Writes a filmstrip PNG and a GIF to out/ (l0_neural.* with
--neural).
"""

import argparse
import time
from pathlib import Path

import torch

from smdp.grid import Grid, make_state
from smdp.mdp import Params
from smdp.neural import load_levels, q_table
from smdp.render import render_text, save_filmstrip, save_gif
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, value_iterate_l0

FLOWER, TREE = (0, 1), (0, 3)
GOALS = (FLOWER, TREE)  # goal of agent 0 (yellow), agent 1 (red)
START = make_state((4, 0), (4, 4), (2, 3), (2, 1))  # each can starts on the far side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="for breaking ties between actions")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--neural", type=Path, help="directory of trained networks (run_neural.py)")
    args = parser.parse_args()

    grid, params = Grid(n_side=5), Params()
    if args.neural:
        device = "cuda" if torch.cuda.is_available() else None
        q0 = load_levels(args.neural, [0], params, GOALS, device)[0]
        q_tables = [q_table(q0, grid, p, goal=GOALS[p]).cpu() for p in (0, 1)]
    else:
        T = Transitions(grid)
        q_tables = []
        for p in (0, 1):
            start = time.perf_counter()
            solution = value_iterate_l0(T, p, GOALS[p], params)
            print(
                f"Agent {p}: {solution.iterations} sweeps, residual {solution.residual:.1e}, "
                f"{time.perf_counter() - start:.2f}s"
            )
            q_tables.append(solution.q)

    greedy = greedy_chooser(grid, q_tables, args.seed)
    states, titles = rollout(grid, START, greedy, args.steps)
    for s, title in zip(states, titles):
        print(title.replace("\n", "  "))
        print(render_text(grid, s, GOALS), end="\n\n")

    args.out.mkdir(exist_ok=True)
    name = "l0_neural" if args.neural else "l0"
    save_filmstrip(grid, states, titles, args.out / f"{name}.png", landmarks=GOALS)
    save_gif(grid, states, titles, args.out / f"{name}.gif", landmarks=GOALS)
    print(f"Saved {args.out / name}.png and .gif")


if __name__ == "__main__":
    main()
