"""Level-1 demo: the red robot has a social goal toward the yellow robot.

    python run_l1.py                          # crossing start, chi = -1, 0, 1
    python run_l1.py --scenario blocking
    python run_l1.py --chis 2 -2 --tau 20
    python run_l1.py --neural out/neural      # the level-0 and level-1 networks instead

Yellow is a level-0 agent (flower). Red is a level-1 agent (tree) with social goal chi: it
weighs yellow's reward by chi and predicts yellow's moves with yellow's level-0 softmax policy.
Red is told yellow's goal; inferring it comes in step 6. Both act greedily.

Scenarios:
  crossing  The run_l0.py start: each can starts on the far side from its goal. Yellow can
            manage alone, so helping looks the same as chi = 0; hindering is clearly visible.
  blocking  Red's can is already on the tree, and red stands right below the yellow can, the
            only cell from which it can be pushed up to the flower. With chi = 0 red has
            nothing to gain by moving and stays in yellow's way; with chi = 1 it pushes the
            yellow can onto the flower itself.

Writes out/l1_<scenario>_chi<chi>_tau<tau>.png and a GIF for each chi, with a _neural suffix
for --neural. The networks were trained for one tau, so --tau must match it then.
"""

import argparse
import time
from pathlib import Path

import torch

from run_l0 import GOALS, START
from smdp.grid import Grid, make_state
from smdp.mdp import Params
from smdp.neural import load_levels, q_table
from smdp.render import render_text, save_filmstrip, save_gif
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, level0_policy_mixture, value_iterate_l0, value_iterate_l1

YELLOW, RED = 0, 1
FLOWER, TREE = GOALS
SCENARIOS = {
    "crossing": START,
    "blocking": make_state((4, 4), (3, 1), (2, 1), TREE),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="crossing")
    parser.add_argument("--chis", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--tau", type=float, default=Params.tau, help="softmax temperature")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="for breaking ties between actions")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--neural", type=Path, help="directory of trained networks (run_neural.py)")
    args = parser.parse_args()

    grid, params = Grid(n_side=5), Params(tau=args.tau)
    belief = [(FLOWER, 1.0)]  # red knows yellow's goal
    if args.neural:
        device = "cuda" if torch.cuda.is_available() else None
        nets = load_levels(args.neural, [0, 1], params, GOALS, device)
        yellow_q = q_table(nets[0], grid, YELLOW, goal=FLOWER).cpu()
    else:
        T = Transitions(grid)
        yellow_q = value_iterate_l0(T, YELLOW, FLOWER, params).q
        yellow_model = level0_policy_mixture(T, YELLOW, belief, params)
    args.out.mkdir(exist_ok=True)

    for chi in args.chis:
        start = time.perf_counter()
        if args.neural:
            red_q = q_table(nets[1], grid, RED, goal=TREE, chi=chi, belief=[1.0, 0.0]).cpu()
            print(f"chi = {chi:g}: network, {time.perf_counter() - start:.1f}s")
        else:
            red = value_iterate_l1(T, RED, TREE, chi, belief, yellow_model, params)
            red_q = red.q
            print(f"chi = {chi:g}: {red.iterations} sweeps, {time.perf_counter() - start:.1f}s")

        greedy = greedy_chooser(grid, [yellow_q, red_q], args.seed)
        states, titles = rollout(grid, SCENARIOS[args.scenario], greedy, args.steps)
        on_flower = [t for t, s in enumerate(states) if (s[4], s[5]) == FLOWER]
        print(f"  yellow can first on the flower at t = {on_flower[0] if on_flower else 'never'}")
        print(render_text(grid, states[-1], GOALS), end="\n\n")

        name = f"l1_{args.scenario}_chi{chi:g}_tau{args.tau:g}" + ("_neural" if args.neural else "")
        titles = [f"red χ = {chi:g}{' (network)' if args.neural else ''}   " + title for title in titles]
        save_filmstrip(grid, states, titles, args.out / f"{name}.png", landmarks=GOALS)
        save_gif(grid, states, titles, args.out / f"{name}.gif", landmarks=GOALS)
    print(f"Saved l1_{args.scenario}_chi*.png and .gif to {args.out}")


if __name__ == "__main__":
    main()
