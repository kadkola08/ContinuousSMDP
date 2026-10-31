"""Value-function figures for the level-0 scenario in run_l0.py.

    python plot_values.py
    python plot_values.py --neural out/neural   # the level-0 network, compared with the table

Writes to out/:
  values_agent0.png, values_agent1.png
      V (with best-action glyphs) and Q (per-action triangles) for one agent at the start
      state. The other robot and both cans stay where they are; the agent is placed on each
      cell. The outlined cell is where the agent actually is.
  values_rollout.gif
      The rollout from run_l0.py, with both agents' V slices redrawn at every step.
With --neural, instead:
  values_agent0_neural.png, values_agent1_neural.png
      The same slices from value iteration (top) and the network (bottom), on one color
      scale. The title says on how many cells the two agree on the best action.
  values_rollout_neural.gif
      The rollout with both agents acting on the network, and the network's V slices.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable

from run_l0 import GOALS, START
from smdp.grid import Grid
from smdp.mdp import Params
from smdp.neural import as_solution, load_levels, q_table
from smdp.render import draw_state, load_images
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, value_iterate_l0
from smdp.valueplot import VALUE_CMAP, agent_slice, draw_q, draw_values, value_norm

AGENT_NAMES = ("Yellow", "Red")


def add_colorbar(fig, axes, norm, extend="neither"):
    mappable = ScalarMappable(norm=norm, cmap=VALUE_CMAP)
    bar = fig.colorbar(mappable, ax=axes, shrink=0.8, pad=0.02, extend=extend)
    bar.set_label("value", color="#1f1f1f")
    bar.outline.set_visible(False)


def save_start_figure(grid, solutions, p, images, path):
    V, Q = agent_slice(grid, solutions[p], START, p)
    # Scale colors to V; a few bad moves have much lower Q and would wash out the rest.
    norm = value_norm(V)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    name = AGENT_NAMES[p]
    draw_values(axes[0], START, p, V, Q, norm, images, GOALS, f"{name} robot: V(s) and best action")
    draw_q(axes[1], START, p, Q, norm, images, GOALS, f"{name} robot: Q(s, a)")
    add_colorbar(fig, axes, norm, extend="min")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def best_action_agreement(Q_tab, Q_net) -> tuple[int, int]:
    """Cells where the network's best action is also best (within 1e-3) in the table."""
    cells = ~np.isnan(Q_tab).any(axis=2)
    best = np.nanmax(Q_tab, axis=2, keepdims=True)
    chosen = np.argmax(np.nan_to_num(Q_net, nan=-np.inf), axis=2)[..., None]
    agree = np.take_along_axis(Q_tab >= best - 1e-3, chosen, axis=2)[..., 0]
    return int((agree & cells).sum()), int(cells.sum())


def save_compare_figure(grid, tabular, network, p, images, path):
    """V and Q at the start state: value iteration on top, the network below."""
    slices = {"value iteration": agent_slice(grid, tabular[p], START, p),
              "network": agent_slice(grid, network[p], START, p)}
    norm = value_norm(*(V for V, _ in slices.values()))
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    name = AGENT_NAMES[p]
    for row, (planner, (V, Q)) in zip(axes, slices.items()):
        draw_values(row[0], START, p, V, Q, norm, images, GOALS, f"{name} robot, {planner}: V(s) and best action")
        draw_q(row[1], START, p, Q, norm, images, GOALS, f"{name} robot, {planner}: Q(s, a)")
    agree, cells = best_action_agreement(slices["value iteration"][1], slices["network"][1])
    fig.suptitle(f"{name} robot, level 0: the network picks a best action on {agree} of {cells} cells",
                 x=0.02, ha="left", color="#1f1f1f")
    add_colorbar(fig, axes, norm, extend="min")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_rollout_gif(grid, solutions, states, titles, images, path):
    slices = [[agent_slice(grid, solutions[p], s, p) for p in (0, 1)] for s in states]
    norm = value_norm(*(V for frame in slices for V, _ in frame))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    fig.subplots_adjust(left=0.01, right=0.97, top=0.9, bottom=0.03, wspace=0.06)
    add_colorbar(fig, axes[1:], norm)

    def update(i):
        draw_state(axes[0], grid, states[i], images, titles[i].replace("\n", "   "), GOALS)
        for p in (0, 1):
            V, Q = slices[i][p]
            draw_values(axes[p + 1], states[i], p, V, Q, norm, images, GOALS, f"{AGENT_NAMES[p]} robot: V(s)")

    animation = FuncAnimation(fig, update, frames=len(states))
    animation.save(path, writer=PillowWriter(fps=1))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="for breaking ties between actions")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--neural", type=Path, help="directory of trained networks (run_neural.py)")
    args = parser.parse_args()

    grid, params = Grid(n_side=5), Params()
    T = Transitions(grid)
    solutions = [value_iterate_l0(T, p, GOALS[p], params) for p in (0, 1)]
    images = load_images()
    args.out.mkdir(exist_ok=True)

    if args.neural:
        device = "cuda" if torch.cuda.is_available() else None
        q0 = load_levels(args.neural, [0], params, GOALS, device)[0]
        network = [as_solution(q_table(q0, grid, p, goal=GOALS[p]).cpu()) for p in (0, 1)]
        for p in (0, 1):
            save_compare_figure(grid, solutions, network, p, images, args.out / f"values_agent{p}_neural.png")
        solutions, suffix = network, "_neural"
    else:
        for p in (0, 1):
            save_start_figure(grid, solutions, p, images, args.out / f"values_agent{p}.png")
        suffix = ""

    greedy = greedy_chooser(grid, [solution.q for solution in solutions], args.seed)
    states, titles = rollout(grid, START, greedy, args.steps)
    save_rollout_gif(grid, solutions, states, titles, images, args.out / f"values_rollout{suffix}.gif")
    print(f"Saved values_agent0{suffix}.png, values_agent1{suffix}.png and values_rollout{suffix}.gif to {args.out}")


if __name__ == "__main__":
    main()
