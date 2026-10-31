"""Runs one experiment from a config file (Algorithm 1: plan, act, observe, repeat).

    python run.py configs/blocking_help.json
    python run.py configs/crossing.json --device cpu
    python run.py configs/blocking_help.json --neural out/neural   # networks, not value iteration

Agents infer each other's goals while acting; nobody is told anything about the other agent.
Writes to out/:
  records/<name>.json   every step: state, actions, action probabilities, and beliefs
  <name>.png            filmstrip of the episode
  <name>.gif            the episode as an animation
  <name>_beliefs.png    how each agent's beliefs about the other evolved
With --neural, agents plan with networks trained by run_neural.py and <name> gets a _neural
suffix, so the two planners' outputs can be compared side by side.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from smdp.config import Config
from smdp.episode import run_episode
from smdp.render import save_filmstrip, save_gif

AGENT_NAMES = ("yellow", "red")
INK = "#1f1f1f"
LINE = "#2a78d6"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("time step", color=INK)
    ax.grid(axis="y", color="#e6e6e6")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_belief_plot(config: Config, record: dict, path):
    """One panel per inferred quantity: P(other's true goal), and E[chi] for level-2 agents."""
    panels = []
    for p, agent in enumerate(config.agents):
        j = 1 - p
        if agent.level >= 1:
            panels.append((p, "goal"))
        if agent.level == 2:
            panels.append((p, "chi"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    ts = [step["t"] for step in record["steps"]]
    for ax, (p, kind) in zip(axes[0], panels):
        j = 1 - p
        beliefs = [step["beliefs"][p] for step in record["steps"]]
        if kind == "goal":
            true_goal = str(config.agents[j].goal)
            values = [b["other_goal"][true_goal] for b in beliefs]
            ax.plot(ts, values, color=LINE, lw=2, marker="o", ms=3)
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel(f"P({AGENT_NAMES[j]}'s true goal)", color=INK)
            style(ax, f"{AGENT_NAMES[p]} infers {AGENT_NAMES[j]}'s physical goal")
        else:
            values = [b["other_expected_chi"] for b in beliefs]
            truth = config.agents[j].chi
            ax.axhline(truth, color=LINE, lw=1, ls=(0, (2, 3)))
            ax.annotate(f"true χ = {truth:+g}", (ts[-1], truth), xytext=(0, 5),
                        textcoords="offset points", ha="right", fontsize=8, color=INK)
            ax.plot(ts, values, color=LINE, lw=2, marker="o", ms=3)
            lo, hi = min(config.social_goals), max(config.social_goals)
            ax.set_ylim(lo - 0.1, hi + 0.1)
            ax.set_ylabel(f"E[{AGENT_NAMES[j]}'s χ]", color=INK)
            style(ax, f"{AGENT_NAMES[p]} infers {AGENT_NAMES[j]}'s social goal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="for breaking ties between actions")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--neural", type=Path, help="directory of trained networks (run_neural.py)")
    args = parser.parse_args()

    config = Config.load(args.config)
    name = config.name + ("_neural" if args.neural else "")
    levels = ", ".join(
        f"{AGENT_NAMES[p]}: level {a.level}, χ = {a.chi:g}, goal {config.landmarks[a.goal]}"
        for p, a in enumerate(config.agents)
    )
    print(f"{config.name} on {args.device} ({levels})")
    start = time.perf_counter()
    record = run_episode(config, device=args.device, seed=args.seed, neural=args.neural)
    print(f"Finished in {time.perf_counter() - start:.0f}s with {record['total_solves']} solves")

    (args.out / "records").mkdir(parents=True, exist_ok=True)
    (args.out / "records" / f"{name}.json").write_text(json.dumps(record, indent=1))
    states = [tuple(step["state"]) for step in record["steps"]]
    titles = [step["title"] for step in record["steps"]]
    save_filmstrip(config.grid, states, titles, args.out / f"{name}.png", landmarks=config.landmarks)
    save_gif(config.grid, states, titles, args.out / f"{name}.gif", landmarks=config.landmarks)
    save_belief_plot(config, record, args.out / f"{name}_beliefs.png")
    print(f"Saved records/{name}.json, {name}.png, {name}.gif and {name}_beliefs.png to {args.out}")


if __name__ == "__main__":
    main()
