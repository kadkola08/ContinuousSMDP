"""Regenerates every figure and GIF, from value iteration and from the trained networks.

    python make_figures.py                          # everything, into figures/
    python make_figures.py --only neural            # just the network figures
    python make_figures.py --out figures_v2 --neural-dir out/neural

Runs each demo script in turn, with its --out pointing into a fresh folder:

    <out>/tabular/   value iteration
    <out>/neural/    the networks in --neural-dir (train them first with run_neural.py)

Figures that don't depend on the planner (rollout.py, plot_policy.py, run_inference.py) go
into tabular/. A script that fails is reported at the end and the rest still run. The tabular
episodes take most of the time, around 15 minutes in all.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
CONFIGS = ("blocking_help", "blocking_hinder", "crossing")
SCENARIOS = ("crossing", "blocking")


def jobs(neural_dir: Path) -> dict:
    """{planner: [script arguments]}; --out is added per planner."""
    shared = [
        ["run_l0.py"],
        *(["run_l1.py", "--scenario", s] for s in SCENARIOS),
        *(["run_l2.py", "--scenario", s] for s in SCENARIOS),
        ["plot_values.py"],
        ["run_social_inference.py"],
        *(["run.py", f"configs/{c}.json"] for c in CONFIGS),
    ]
    tabular = [
        ["rollout.py"],
        *(["rollout.py", "--random", "--seed", str(seed)] for seed in (0, 3)),
        ["plot_policy.py"],
        ["run_inference.py"],
    ]
    return {
        "tabular": shared + tabular,
        "neural": [args + ["--neural", str(neural_dir)] for args in shared],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    parser.add_argument("--neural-dir", type=Path, default=HERE / "out" / "neural")
    parser.add_argument("--only", choices=("tabular", "neural"), help="make one planner's figures")
    args = parser.parse_args()

    planners = [args.only] if args.only else ["neural", "tabular"]  # the fast ones first
    if "neural" in planners:
        missing = [f"q{level}.pt" for level in range(3) if not (args.neural_dir / f"q{level}.pt").exists()]
        if missing:
            raise SystemExit(f"{args.neural_dir} has no {', '.join(missing)}; train them with run_neural.py")

    failed = []
    start = time.perf_counter()
    for planner in planners:
        out = (args.out / planner).resolve()
        out.mkdir(parents=True, exist_ok=True)
        for script in jobs(args.neural_dir.resolve())[planner]:
            command = [sys.executable, *script, "--out", str(out)]
            print(f"[{planner}] {' '.join(script)}", flush=True)
            job_start = time.perf_counter()
            result = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
            if result.returncode != 0:
                failed.append((planner, script, result.stderr.strip().splitlines()[-1:]))
                print(f"    FAILED: {' '.join(failed[-1][2])}", flush=True)
            else:
                print(f"    done in {time.perf_counter() - job_start:.0f}s", flush=True)

    print(f"\nFinished in {time.perf_counter() - start:.0f}s; figures in {args.out.resolve()}")
    for planner, script, error in failed:
        print(f"  failed [{planner}] {' '.join(script)}: {' '.join(error)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
