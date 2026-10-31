"""The simulation loop (Algorithm 1): plan, act, observe, repeat.

Within a time step agent 0 moves first, then agent 1. Each agent plans right before its move,
and both agents observe every move, so each always acts on its freshest beliefs:

    agent 0 plans, moves s -> s1;  both observe it
    agent 1 plans, moves s1 -> s2; both observe it

The record is a JSON-serializable dict. Goals in beliefs are given as landmark indices.
"""

import time

from smdp.agents import Models, make_agents
from smdp.config import Config
from smdp.neural_agents import NeuralModels, make_neural_agents
from smdp.render import is_blocked, step_title
from smdp.solver import Transitions


def _jsonable_beliefs(beliefs: dict, landmarks) -> dict:
    out = {}
    for name, value in beliefs.items():
        if name in ("other_goal", "about_me"):
            out[name] = {str(landmarks.index(g)): p for g, p in value.items()}
        elif name == "other_chi":
            out[name] = {f"{chi:g}": p for chi, p in value.items()}
        else:
            out[name] = value
    return out


def run_episode(config: Config, device=None, seed: int = 0, log=print, neural=None) -> dict:
    """``neural`` is a directory of networks trained by run_neural.py; agents then plan with
    them (smdp.neural_agents) instead of value iteration."""
    start = time.perf_counter()
    if neural is not None:
        models = NeuralModels.load(neural, config, device)
        agents = make_neural_agents(config, models, seed)
    else:
        models = Models(Transitions(config.grid, device), config)
        agents = make_agents(config, models, seed)
    log(f"Set up {len(agents)} agents ({models.solves} solves, {time.perf_counter() - start:.1f}s)")

    def snapshot():
        return [_jsonable_beliefs(agent.beliefs(), config.landmarks) for agent in agents]

    s = config.initial_state
    record = {
        "config": config.name,
        "planner": "value iteration" if neural is None else f"networks in {neural}",
        "landmarks": [list(g) for g in config.landmarks],
        "agents": [
            {"goal": a.goal, "level": a.level, "chi": a.chi} for a in config.agents
        ],
        "steps": [
            {"t": 0, "state": list(s), "title": step_title(0, None, (False, False)), "beliefs": snapshot()}
        ],
    }
    for t in range(1, config.steps + 1):
        step_start, solves_before = time.perf_counter(), models.solves
        moves, probs, half = [], [], s
        for agent in agents:
            agent.plan()
            probs.append(agent.action_probs(half))
            action = agent.act(half)
            after = config.grid.step(half, agent.p, action)
            for observer in agents:
                if observer is agent:
                    observer.observe_own(half, after)
                else:
                    observer.observe_other(half, after)
            moves.append((half, action, after))
            half = after
        (s0, a, s1), (_, b, s2) = moves
        blocked = (is_blocked(s0, s1, a), is_blocked(s1, s2, b))
        record["steps"].append(
            {
                "t": t,
                "state": list(s2),
                "half_state": list(s1),
                "actions": [a, b],
                "action_probs": probs,
                "title": step_title(t, (a, b), blocked),
                "beliefs": snapshot(),
            }
        )
        log(
            f"t = {t:2d}: {step_title(t, (a, b), blocked).split(chr(10), 1)[1].replace(chr(10), ', ')}"
            f"  ({models.solves - solves_before} solves, {time.perf_counter() - step_start:.1f}s)"
        )
        s = s2
    record["total_solves"] = models.solves
    return record
