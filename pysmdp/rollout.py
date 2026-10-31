"""Visualizes a rollout of the grid world.

    python rollout.py            # scripted moves that exercise every push rule
    python rollout.py --random   # random moves for both agents

Writes a filmstrip PNG and an animated GIF to out/ and prints each step as text.
"""

import argparse
import random
from pathlib import Path

from smdp.grid import DOWN, LEFT, N_ACTIONS, RIGHT, STAY, UP, Grid, make_state
from smdp.render import render_text, save_filmstrip, save_gif
from smdp.simulate import rollout

# Joint moves (yellow, red) for the scripted demo, starting from SCRIPTED_START on a 5x5 grid.
SCRIPTED_START = make_state((2, 0), (0, 4), (2, 1), (2, 2))
SCRIPTED_MOVES = [
    (RIGHT, DOWN),  # yellow chain-pushes both cans
    (RIGHT, LEFT),  # again, the red can reaches the wall
    (RIGHT, STAY),  # chain push into the wall: nothing moves
    (UP, STAY),
    (RIGHT, STAY),  # red robot is in the way: yellow is blocked
    (STAY, DOWN),  # red pushes yellow's can
    (STAY, RIGHT),  # red pushes its own can into the wall: blocked
    (STAY, DOWN),  # red pushes yellow's can again
    (LEFT, LEFT),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="random moves instead of the script")
    parser.add_argument("--steps", type=int, default=14, help="steps for --random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()

    grid = Grid(n_side=5)
    if args.random:
        rng = random.Random(args.seed)
        moves = [(rng.randrange(N_ACTIONS), rng.randrange(N_ACTIONS)) for _ in range(args.steps)]
        name = f"rollout_random_seed{args.seed}"
    else:
        moves = SCRIPTED_MOVES
        name = "rollout_scripted"

    states, titles = rollout(grid, SCRIPTED_START, lambda t, s, p: moves[t - 1][p], len(moves))
    for s, title in zip(states, titles):
        print(title.replace("\n", "  "))
        print(render_text(grid, s), end="\n\n")

    args.out.mkdir(exist_ok=True)
    save_filmstrip(grid, states, titles, args.out / f"{name}.png")
    save_gif(grid, states, titles, args.out / f"{name}.gif")
    print(f"Saved {args.out / name}.png and .gif")


if __name__ == "__main__":
    main()
