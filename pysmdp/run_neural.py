"""Trains the level 0, 1 and 2 Q-networks on the grid and checks them against value iteration.

    python run_neural.py                      # train all levels, 20000 steps each
    python run_neural.py --levels 2 --load    # reuse saved level 0 and 1, train level 2
    python run_neural.py --load --steps 0     # only evaluate saved networks

Each level is trained by fitted value iteration (smdp.neural) with the level below frozen as
the other agent's model, so errors compound upwards exactly as they would without a table to
fall back on. The networks never see a table; the tabular solutions are only used to score
them, on every valid state, for a fixed set of hypotheses:

    q_mae    mean |Q_net - Q_tab|
    adv_mae  the same for Q minus its mean over actions, which is all a softmax policy sees
    greedy   fraction of states where the network's best action is also best in the table
    kl, tv   KL divergence and total variation between the two softmax policies (Eq. 7),
             which is what goal inference uses as likelihoods

After level 1, yellow infers red's (goal, chi) along the blocking scenario twice, with the
tabular and with the network level-1 policies as likelihoods, as in run_social_inference.py.

Networks are saved to out/neural/q<level>.pt and the scores to out/neural/report.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from run_l1 import FLOWER, SCENARIOS, TREE
from smdp.grid import Grid
from smdp.inference import CHIS, GoalEstimator
from smdp.mdp import Params
from smdp.neural import NeuralQ, compare_q, fit, q_table
from smdp.policy import Policy
from smdp.simulate import greedy_chooser, rollout
from smdp.solver import (
    Transitions,
    level0_policy_mixture,
    level1_policies,
    value_iterate_l0,
    value_iterate_l1,
    value_iterate_l2,
)

LANDMARKS = [FLOWER, TREE]
YELLOW, RED = 0, 1


def one_hot(index, size):
    return [1.0 if i == index else 0.0 for i in range(size)]


def as_belief(probs):
    """A probability vector over LANDMARKS as the solver's (goal, prob) pairs."""
    return [(g, p) for g, p in zip(LANDMARKS, probs) if p > 0]


def joint_belief(weights: dict):
    """{(goal, chi): prob} as the flat vector over (landmark, chi) the networks take."""
    return [weights.get((g, c), 0.0) for g in LANDMARKS for c in CHIS]


def tabular_cases(T, params):
    """(level, agent, name, hypothesis fields, tabular Q) for the evaluation set."""
    cases = []
    for p, goal in ((YELLOW, FLOWER), (RED, TREE)):
        cases.append((0, p, f"agent {p} -> {goal}", {"goal": goal}, value_iterate_l0(T, p, goal, params).q))
    for p, goal, chi, belief in (
        (RED, TREE, 1.0, [1.0, 0.0]),
        (RED, TREE, 0.0, [1.0, 0.0]),
        (RED, TREE, -1.0, [1.0, 0.0]),
        (RED, TREE, 2.0, [0.5, 0.5]),
        (YELLOW, FLOWER, -0.5, [0.3, 0.7]),
    ):
        other = level0_policy_mixture(T, 1 - p, as_belief(belief), params)
        q = value_iterate_l1(T, p, goal, chi, as_belief(belief), other, params).q
        cases.append((1, p, f"agent {p} chi {chi:+g} b {belief}", {"goal": goal, "chi": chi, "belief": belief}, q))
    for weights, about_me in (
        ({(TREE, 1.0): 1.0}, [1.0, 0.0]),
        ({(TREE, -1.0): 1.0}, [1.0, 0.0]),
        ({(TREE, 1.0): 0.5, (TREE, -1.0): 0.3, (FLOWER, 0.0): 0.2}, [0.7, 0.3]),
    ):
        about_yellow = as_belief(about_me)
        yellow_model = level0_policy_mixture(T, YELLOW, about_yellow, params)
        red = {
            h: Policy.from_q(RED, value_iterate_l1(T, RED, h[0], h[1], about_yellow, yellow_model, params).q, params.tau)
            for h in weights
        }
        red_model = Policy.mixture([(w, red[h]) for h, w in weights.items()])
        q = value_iterate_l2(T, YELLOW, FLOWER, 0.0, list(weights.items()), about_yellow, red_model, params).q
        name = "yellow, red ~ " + ", ".join(f"{w:g}x({'tree' if g == TREE else 'flower'},{c:+g})" for (g, c), w in weights.items())
        fields = {"goal": FLOWER, "chi": 0.0, "joint": joint_belief(weights), "about_me": about_me}
        cases.append((2, YELLOW, name, fields, q))
    return cases


def network_q(model: NeuralQ, T, p, fields):
    """The network's Q for agent p on every valid state, as (n_valid, n_actions)."""
    return q_table(model, T.grid, p, **fields)[:, T.valid].T


def scores(model, cases, T, params):
    with torch.no_grad():
        return [
            (name, compare_q(network_q(model, T, p, fields), q[:, T.valid].T, params.tau))
            for level, p, name, fields, q in cases
            if level == model.level
        ]


def summary(results) -> str:
    mean = {k: sum(r[k] for _, r in results) / len(results) for k in results[0][1]}
    return f"q_mae {mean['q_mae']:.3f}  adv_mae {mean['adv_mae']:.4f}  greedy {mean['greedy']:.3f}  tv {mean['tv']:.4f}"


def inference_check(T, q1: NeuralQ, params, steps=12):
    """Yellow's belief over red's (goal, chi) with tabular vs network likelihoods."""
    about_yellow = [(FLOWER, 1.0)]
    yellow_model = level0_policy_mixture(T, YELLOW, about_yellow, params)
    tabular = level1_policies(T, RED, LANDMARKS, CHIS, about_yellow, yellow_model, params)
    network = {
        (g, c): Policy.from_q(RED, q_table(q1, T.grid, RED, goal=g, chi=c, belief=[1.0, 0.0]), params.tau)
        for g, c in tabular
    }
    yellow_q = value_iterate_l0(T, YELLOW, FLOWER, params).q
    report = {}
    for true_chi in (1.0, 0.0, -1.0):
        red_q = torch.log(tabular[(TREE, true_chi)].probs)
        halves = []
        states, _ = rollout(T.grid, SCENARIOS["blocking"], greedy_chooser(T.grid, [yellow_q, red_q]), steps, halves)
        traces = {}
        for name, policies in (("tabular", tabular), ("network", network)):
            estimator = GoalEstimator(T.grid, policies)
            expected = []
            for before, after in zip(halves, states[1:]):
                estimator.update(before, after)
                expected.append(sum(c * p for c, p in estimator.marginal(1).items()))
            traces[name] = (expected, estimator.probs)
        gap = max(abs(a - b) for a, b in zip(traces["tabular"][0], traces["network"][0]))
        tv = 0.5 * sum(abs(traces["tabular"][1][h] - traces["network"][1][h]) for h in tabular)
        print(f"  red chi {true_chi:+g}: final E[chi] tabular {traces['tabular'][0][-1]:+.2f}, "
              f"network {traces['network'][0][-1]:+.2f}; max |gap| {gap:.3f}, final posterior TV {tv:.3f}")
        report[f"{true_chi:g}"] = {
            "expected_chi": {k: v[0] for k, v in traces.items()}, "max_gap": gap, "final_tv": tv,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--load", action="store_true", help="start from saved networks where they exist")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "neural")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else None
    grid, params = Grid(n_side=5), Params()
    T = Transitions(grid, device)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cases = tabular_cases(T, params)
    print(f"Solved {len(cases)} tabular evaluation cases in {time.perf_counter() - start:.0f}s")

    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    models = {}
    for level in range(max(args.levels) + 1):
        path = args.out / f"q{level}.pt"
        train = level in args.levels and args.steps > 0
        if path.exists() and (args.load or not train):
            model = NeuralQ.load(path, device)
        elif train:
            model = NeuralQ(level, grid.n_side, LANDMARKS, CHIS, params, args.width, args.depth).to(device)
        else:
            raise SystemExit(f"{path} is missing; train level {level} first")
        if train:
            print(f"Training level {level} ({sum(p.numel() for p in model.parameters())} parameters)")
            start = time.perf_counter()
            fit(model, models.get(level - 1), grid, steps=args.steps, batch=args.batch, lr=args.lr,
                evaluate=lambda: summary(scores(model, cases, T, params)))
            print(f"  trained in {time.perf_counter() - start:.0f}s")
            model.save(path)
        models[level] = model.eval()

        if level in args.levels:
            print(f"Level {level} against value iteration:")
            results = scores(model, cases, T, params)
            for name, r in results:
                print(f"  {name:55s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
            report[f"level{level}"] = dict(results)
            if level == 1:
                print("Goal inference with level-1 likelihoods (blocking scenario):")
                report["inference"] = inference_check(T, model, params)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved networks and {report_path}")


if __name__ == "__main__":
    main()
