"""Social goal inference demo (Eq. 5).

    python run_social_inference.py                       # blocking scenario
    python run_social_inference.py --scenario crossing
    python run_social_inference.py --neural out/neural   # also with network likelihoods

Red is a level-1 agent with a true social goal chi (one run each for chi = 1, 0, -1); yellow
is a level-0 agent heading for the flower. Yellow watches red and keeps a joint belief over
red's physical goal (flower or tree) and social goal (the paper's 7 chi values), using red's
level-1 policy for each of the 14 hypotheses. Red is told yellow's goal, as in run_l1.py.

With --neural, yellow runs the same inference a second time along the same trajectories, with
the level-1 network's policies as likelihoods instead of the tabular ones. Red still acts with
its tabular policy, so the two beliefs differ only through the likelihoods.

Writes to out/ (with a _neural suffix for --neural):
  social_inference_<scenario>.png         yellow's expected chi over time (dotted lines are the
                                          true values) and its final belief over chi in each run;
                                          with --neural, dashed lines and hatched bars are the
                                          network's
  social_inference_<scenario>_chi<c>.gif  the run for true chi = c, with yellow's belief
                                          evolving next to it
Solving the 14 level-1 policies takes about a minute.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from run_l1 import FLOWER, SCENARIOS, TREE
from smdp.grid import Grid
from smdp.inference import CHIS, GoalEstimator
from smdp.mdp import Params
from smdp.neural import load_levels, q_table
from smdp.policy import Policy
from smdp.render import draw_state, load_images
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import Transitions, level0_policy_mixture, level1_policies, value_iterate_l0

TRUE_CHIS = (1.0, 0.0, -1.0)
# Diverging encoding for the sign of chi: blue = helping, gray = neutral, red = hindering.
CHI_COLORS = {1.0: "#2a78d6", 0.0: "#8a8984", -1.0: "#e34948"}
INK, MUTED = "#1f1f1f", "#6b6b6b"
LANDMARKS = (FLOWER, TREE)
LINESTYLES = {"tabular": "-", "network": (0, (4, 2))}


def episode(T, hypotheses, yellow_q, start, true_chi, steps):
    """Red acts greedily with true_chi. Returns the states, and the state before each red move."""
    red_q = torch.log(hypotheses[(TREE, true_chi)].probs)  # same argmax as red's Q
    halves = []
    states, _ = rollout(T.grid, start, greedy_chooser(T.grid, [yellow_q, red_q]), steps, halves)
    return states, halves


def infer(grid, hypotheses, states, halves):
    """Yellow's E[chi] and P(chi) after each step, and its final P(goal)."""
    estimator = GoalEstimator(grid, hypotheses)
    p_chi = [estimator.marginal(1)]
    for before, after in zip(halves, states[1:]):  # red moves from halves[t] to states[t + 1]
        estimator.update(before, after)
        p_chi.append(estimator.marginal(1))
    expected = [sum(c * p for c, p in dist.items()) for dist in p_chi]
    return expected, p_chi, estimator.marginal(0)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def draw_bars(ax, beliefs: dict, color, t=None):
    """P(chi) per planner; the network's bars are hatched and sit right of the tabular ones."""
    width = 0.8 / len(beliefs)
    for i, (planner, p_chi) in enumerate(beliefs.items()):
        offsets = [k + (i - (len(beliefs) - 1) / 2) * width for k in range(len(CHIS))]
        hatched = planner == "network"
        ax.bar(offsets, [p_chi[c] for c in CHIS], width=width * 0.92, label=planner,
               color="white" if hatched else color, edgecolor=color, hatch="////" if hatched else None)
    ax.set_xticks(range(len(CHIS)), [f"{c:+g}" for c in CHIS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("χ", color=INK)
    ax.grid(axis="y", color="#e6e6e6")
    ax.set_axisbelow(True)
    style(ax)


def draw_expected(ax, traces: dict, chi, color, upto=None):
    """E[chi] over time per planner, up to step ``upto``."""
    ax.axhline(chi, color=color, lw=1, ls=(0, (2, 3)))
    for planner, expected in traces.items():
        shown = expected if upto is None else expected[: upto + 1]
        label = f"red's true χ = {chi:+g}" + (f" ({planner})" if len(traces) > 1 else "")
        ax.plot(range(len(shown)), shown, color=color, lw=2, marker="o", ms=4,
                ls=LINESTYLES[planner], label=label)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel("time step", color=INK)
    ax.grid(axis="y", color="#e6e6e6")
    style(ax)


def save_summary(results, args, path):
    planners = list(next(iter(results.values())).keys())
    fig = plt.figure(figsize=(12, 8))
    spec = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(spec[0, :])
    for chi, by_planner in results.items():
        color = CHI_COLORS[chi]
        traces = {planner: r["expected"] for planner, r in by_planner.items()}
        draw_expected(ax, traces, chi, color)
        last = traces[planners[0]]
        ax.annotate(f"true χ = {chi:+g}", (len(last) - 1, last[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, args.steps + 2.5)
    ax.set_xticks(range(args.steps + 1))
    ax.set_ylabel("yellow's expected χ for red", color=INK)
    note = "; dashed lines use the network's likelihoods" if "network" in planners else ""
    ax.set_title(f"Yellow infers red's social goal ({args.scenario} scenario); dotted lines are the truth{note}",
                 loc="left", color=INK)
    ax.legend(loc="upper right", frameon=False, ncols=3, fontsize=8 if len(planners) > 1 else 10)

    for col, (chi, by_planner) in enumerate(results.items()):
        bars = fig.add_subplot(spec[1, col])
        draw_bars(bars, {planner: r["p_chi"][-1] for planner, r in by_planner.items()}, CHI_COLORS[chi])
        if col == 0:
            bars.set_ylabel(f"P(χ) at t = {args.steps}", color=INK)
            if len(planners) > 1:
                bars.legend(frameon=False, fontsize=8, loc="upper left")
        bars.set_title(f"red's true χ = {chi:+g}", loc="left", fontsize=10, color=INK)
        truth = CHIS.index(chi)
        top = max(r["p_chi"][-1][chi] for r in by_planner.values())
        bars.annotate("true", (truth, top), xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_inference_gif(grid, states, by_planner, chi, scenario, path):
    """The episode on the left; yellow's E[chi] so far and current P(chi) on the right."""
    color = CHI_COLORS[chi]
    images = load_images()
    fig = plt.figure(figsize=(11, 4.6))
    spec = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.55, wspace=0.2)
    board, curve, bars = fig.add_subplot(spec[:, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 1])
    traces = {planner: r["expected"] for planner, r in by_planner.items()}

    def update(t):
        draw_state(board, grid, states[t], images, f"{scenario}, red's true χ = {chi:+g}, t = {t}", LANDMARKS)
        curve.clear()
        draw_expected(curve, traces, chi, color, upto=t)
        curve.set_xlim(0, len(states) - 1)
        curve.set_ylabel("E[χ]", color=INK)
        curve.set_title("yellow's expected χ for red", loc="left", fontsize=10, color=INK)
        if len(traces) > 1:
            curve.legend(frameon=False, fontsize=7, loc="lower left")
        bars.clear()
        draw_bars(bars, {planner: r["p_chi"][t] for planner, r in by_planner.items()}, color)
        bars.set_ylabel("P(χ)", color=INK)
        if len(traces) > 1:
            bars.legend(frameon=False, fontsize=7, loc="upper left")

    animation = FuncAnimation(fig, update, frames=len(states))
    animation.save(path, writer=PillowWriter(fps=1.5))
    plt.close(fig)


def network_hypotheses(directory, grid, params, keys):
    """Red's level-1 policy for each (goal, chi) from the network, red knowing yellow's goal."""
    device = "cuda" if torch.cuda.is_available() else None
    q1 = load_levels(directory, [1], params, LANDMARKS, device)[1]
    belief = [1.0 if g == FLOWER else 0.0 for g in LANDMARKS]
    return {
        (g, c): Policy.from_q(1, q_table(q1, grid, 1, goal=g, chi=c, belief=belief).cpu(), params.tau)
        for g, c in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="blocking")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--tau", type=float, default=Params.tau)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--neural", type=Path, help="directory of trained networks (run_neural.py)")
    args = parser.parse_args()

    grid, params = Grid(n_side=5), Params(tau=args.tau)
    T = Transitions(grid)
    start = time.perf_counter()
    belief_about_yellow = [(FLOWER, 1.0)]
    yellow_model = level0_policy_mixture(T, 0, belief_about_yellow, params)
    hypotheses = {"tabular": level1_policies(
        T, 1, [FLOWER, TREE], CHIS, belief_about_yellow, yellow_model, params
    )}
    print(f"Solved {len(hypotheses['tabular'])} level-1 policies in {time.perf_counter() - start:.0f}s")
    if args.neural:
        hypotheses["network"] = network_hypotheses(args.neural, grid, params, hypotheses["tabular"])
    yellow = value_iterate_l0(T, 0, FLOWER, params)

    suffix = "_neural" if args.neural else ""
    args.out.mkdir(exist_ok=True)
    results = {}
    for chi in TRUE_CHIS:
        states, halves = episode(T, hypotheses["tabular"], yellow.q, SCENARIOS[args.scenario], chi, args.steps)
        results[chi] = {}
        for planner, policies in hypotheses.items():
            expected, p_chi, p_goal = infer(grid, policies, states, halves)
            results[chi][planner] = {"expected": expected, "p_chi": p_chi}
            print(
                f"true chi {chi:+g} ({planner}): final E[chi] {expected[-1]:+.2f}, "
                f"P(red's goal = tree) {p_goal[TREE]:.2f}"
            )
        save_inference_gif(grid, states, results[chi], chi, args.scenario,
                           args.out / f"social_inference_{args.scenario}_chi{chi:g}{suffix}.gif")

    path = args.out / f"social_inference_{args.scenario}{suffix}.png"
    save_summary(results, args, path)
    print(f"Saved {path} and social_inference_{args.scenario}_chi*{suffix}.gif")


if __name__ == "__main__":
    main()
