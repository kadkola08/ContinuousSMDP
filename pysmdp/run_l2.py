"""Level-2 demo: does reasoning about red's social goal help yellow?

    python run_l2.py                        # blocking scenario
    python run_l2.py --scenario crossing
    python run_l2.py --neural out/neural     # the level 0-2 networks instead of value iteration

Red is a level-1 agent (tree) with social goal chi_red = 1, 0 or -1, and knows yellow wants the
flower. Yellow (flower, no social goal of its own) is run twice against each red:
  level 0  plans as if red moves at random;
  level 2  models red as a level-1 agent. It is told red's goal and chi here; inferring them
           while acting comes in step 9.

Writes out/l2_<scenario>_redchi<chi>.png (level-0 yellow on top, level-2 below) and .gif (the
level-2 run) for each chi, and out/l2_<scenario>_summary.png with yellow's total reward and
number of moves. With --neural every file gets a _neural suffix.
"""

import argparse
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import torch

from run_l1 import FLOWER, SCENARIOS, TREE
from smdp.grid import Grid
from smdp.inference import CHIS
from smdp.mdp import Params, reward
from smdp.neural import load_levels, q_table
from smdp.render import save_comparison, save_gif
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import (
    Transitions,
    level0_policy_mixture,
    level1_policies,
    value_iterate_l0,
    value_iterate_l2,
)

RED_CHIS = (1.0, 0.0, -1.0)
LEVEL_COLORS = {"level 0": "#2a78d6", "level 2": "#eb6834"}  # categorical slots 1 and 2
INK = "#1f1f1f"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="blocking")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tau", type=float, default=Params.tau)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--neural", type=Path, help="directory of trained networks (run_neural.py)")
    args = parser.parse_args()

    grid, params = Grid(n_side=5), Params(tau=args.tau)
    landmarks = (FLOWER, TREE)
    about_yellow = [(FLOWER, 1.0)]  # red's belief about yellow's goal
    if args.neural:
        device = "cuda" if torch.cuda.is_available() else None
        nets = load_levels(args.neural, [0, 1, 2], params, landmarks, device)
        if nets[2].chis_list != list(CHIS):
            raise SystemExit(f"q2.pt was trained with social goals {nets[2].chis_list}, not {list(CHIS)}")
        yellow_l0_q = q_table(nets[0], grid, 0, goal=FLOWER).cpu()
    else:
        T = Transitions(grid)
        yellow_model = level0_policy_mixture(T, 0, about_yellow, params)
        red = level1_policies(T, 1, [TREE], RED_CHIS, about_yellow, yellow_model, params)
        yellow_l0_q = value_iterate_l0(T, 0, FLOWER, params).q
    suffix = "_neural" if args.neural else ""
    start = SCENARIOS[args.scenario]
    args.out.mkdir(exist_ok=True)

    totals, moves = {}, {}
    for chi in RED_CHIS:
        if args.neural:
            red_q = q_table(nets[1], grid, 1, goal=TREE, chi=chi, belief=[1.0, 0.0]).cpu()
            joint = [1.0 if (g, c) == (TREE, chi) else 0.0 for g in landmarks for c in CHIS]
            yellow_l2_q = q_table(
                nets[2], grid, 0, goal=FLOWER, chi=0.0, joint=joint, about_me=[1.0, 0.0]
            ).cpu()
        else:
            red_policy = red[(TREE, chi)]
            red_q = torch.log(red_policy.probs)  # same argmax as red's Q
            yellow_l2_q = value_iterate_l2(
                T, 0, FLOWER, 0.0, [((TREE, chi), 1.0)], about_yellow, red_policy, params
            ).q
        rows = []
        for level, yellow_q in (("level 0", yellow_l0_q), ("level 2", yellow_l2_q)):
            states, titles = rollout(grid, start, greedy_chooser(grid, [yellow_q, red_q]), args.steps)
            totals[(chi, level)] = sum(reward(s, 0, FLOWER, params) for s in states)
            moves[(chi, level)] = sum("Y: stay" not in title for title in titles[1:])
            print(
                f"red chi {chi:+g}, yellow {level}: {moves[(chi, level)]} yellow moves, "
                f"total yellow reward {totals[(chi, level)]:.2f}"
            )
            rows.append((f"yellow\n{level}", states, titles))
        name = f"l2_{args.scenario}_redchi{chi:g}{suffix}"
        save_comparison(grid, rows, args.out / f"{name}.png", landmarks)
        _, states, titles = rows[1]
        titles = [f"yellow level 2, red χ = {chi:+g}\n" + title for title in titles]
        save_gif(grid, states, titles, args.out / f"{name}.gif", landmarks=landmarks)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(RED_CHIS))
    metrics = (
        (totals, f"yellow's total reward over {args.steps} steps", "{:.1f}"),
        (moves, "number of yellow moves", "{:d}"),
    )
    for ax, (values, label, fmt) in zip(axes, metrics):
        for k, (level, color) in enumerate(LEVEL_COLORS.items()):
            heights = [values[(chi, level)] for chi in RED_CHIS]
            # A 2px surface gap between the paired bars.
            bars = ax.bar(x + (k - 0.5) * 0.38, heights, width=0.36, color=color, label=f"yellow {level}")
            ax.bar_label(bars, labels=[fmt.format(h) for h in heights], fontsize=8, color=INK, padding=2)
        ax.set_xticks(x, [f"red's χ = {chi:+g}" for chi in RED_CHIS])
        ax.set_title(label, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left")
    planner = "networks" if args.neural else "value iteration"
    fig.suptitle(f"Level-0 vs level-2 yellow ({args.scenario} scenario, {planner})", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(args.out / f"l2_{args.scenario}_summary{suffix}.png", dpi=110)
    print(f"Saved l2_{args.scenario}_*{suffix}.png and .gif to {args.out}")


if __name__ == "__main__":
    main()
