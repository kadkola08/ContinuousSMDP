"""Softmax policies (Eq. 7) of the yellow robot at several temperatures.

    python plot_policy.py

Same slice as plot_values.py: the red robot and both cans stay where they are in the run_l0.py
start state, and the yellow robot is placed on each cell. Arrow length is the probability of
each move; the dot's area is the probability of staying. Writes out/policy_tau.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from run_l0 import GOALS, START
from smdp.grid import Grid
from smdp.mdp import Params
from smdp.policy import Policy
from smdp.render import load_images
from smdp.solver import Transitions, value_iterate_l0
from smdp.valueplot import draw_policy, policy_slice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=int, default=0)
    parser.add_argument("--taus", type=float, nargs="+", default=[0, 1, 5, 20])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()

    grid, params, p = Grid(n_side=5), Params(), args.agent
    T = Transitions(grid)
    solution = value_iterate_l0(T, p, GOALS[p], params)
    images = load_images()

    fig, axes = plt.subplots(1, len(args.taus), figsize=(4.6 * len(args.taus), 5))
    for ax, tau in zip(axes, args.taus):
        P = policy_slice(grid, Policy.from_q(p, solution.q, tau), START)
        draw_policy(ax, START, p, P, images, GOALS, f"τ = {tau:g}")
    fig.suptitle("Yellow robot's softmax policy ψ(s, a) at increasing τ", x=0.01, ha="left")
    fig.tight_layout()
    args.out.mkdir(exist_ok=True)
    fig.savefig(args.out / "policy_tau.png", dpi=110)
    print(f"Saved {args.out / 'policy_tau.png'}")


if __name__ == "__main__":
    main()
