"""Physical goal inference demo (Eq. 6).

    python run_inference.py

Yellow acts as a level-0 agent from the crossing start, once with the flower as its goal and
once with the tree. Red stays still and watches, updating a belief over yellow's goal after
every move. Writes out/inference.png: the belief over time for both runs, with snapshots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from run_l0 import GOALS, START
from smdp.grid import STAY, Grid
from smdp.inference import GoalEstimator
from smdp.mdp import Params
from smdp.render import draw_state, load_images
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, level0_policies, value_iterate_l0

FLOWER, TREE = GOALS
GOAL_NAMES = {FLOWER: "flower", TREE: "tree"}
SERIES_COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2
INK, MUTED = "#1f1f1f", "#6b6b6b"


def run(T, params, policies, true_goal, steps):
    """Yellow heads for true_goal while red stays; returns states and P(flower) per step."""
    yellow = value_iterate_l0(T, 0, true_goal, params)
    stay = torch.zeros_like(yellow.q)
    stay[STAY] = 1.0
    states, _ = rollout(T.grid, START, greedy_chooser(T.grid, [yellow.q, stay]), steps)
    estimator = GoalEstimator(T.grid, policies)
    p_flower = [estimator.probs[FLOWER]]
    for before, after in zip(states, states[1:]):
        estimator.update(before, after)
        p_flower.append(estimator.probs[FLOWER])
    return states, p_flower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--tau", type=float, default=Params.tau)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 4, 7, 10, 15])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()

    grid, params = Grid(n_side=5), Params(tau=args.tau)
    T = Transitions(grid)
    policies = level0_policies(T, 0, [FLOWER, TREE], params)
    runs = {g: run(T, params, policies, g, args.steps) for g in (FLOWER, TREE)}

    n_snap = len(args.snapshots)
    fig = plt.figure(figsize=(2.4 * n_snap + 1, 10))
    spec = fig.add_gridspec(3, n_snap, height_ratios=[1.5, 1, 1], hspace=0.45)

    ax = fig.add_subplot(spec[0, :])
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 4)))
    for color, (goal, (_, p_flower)) in zip(SERIES_COLORS, runs.items()):
        label = f"true goal: {GOAL_NAMES[goal]}"
        ax.plot(range(len(p_flower)), p_flower, color=color, lw=2, marker="o", ms=4, label=label)
        ax.annotate(
            label, (len(p_flower) - 1, p_flower[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", color=INK, fontsize=9,
        )
    ax.set_xlim(0, args.steps + 3.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("time step", color=INK)
    ax.set_ylabel("red's belief P(yellow's goal = flower)", color=INK)
    ax.set_title("Red infers yellow's physical goal from its moves", loc="left", color=INK)
    ax.legend(loc="lower left", frameon=False)
    ax.set_xticks(range(0, args.steps + 1))
    ax.grid(axis="y", color="#e6e6e6")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    images = load_images()
    for row, (goal, (states, p_flower)) in enumerate(runs.items(), start=1):
        for col, t in enumerate(args.snapshots):
            snap = fig.add_subplot(spec[row, col])
            title = f"{GOAL_NAMES[goal]} run, t = {t}\nP(flower) = {p_flower[t]:.2f}"
            draw_state(snap, grid, states[t], images, title, GOALS)

    args.out.mkdir(exist_ok=True)
    fig.savefig(args.out / "inference.png", dpi=110, bbox_inches="tight")
    print(f"Saved {args.out / 'inference.png'}")


if __name__ == "__main__":
    main()
