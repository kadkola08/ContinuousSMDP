"""Wiring of the network-planning agents, with small untrained networks."""

import math
from pathlib import Path

import pytest
import torch

from smdp.config import Config
from smdp.episode import run_episode
from smdp.neural import NeuralQ
from smdp.neural_agents import NeuralModels

CONFIG = Path(__file__).parent.parent / "configs" / "blocking_help.json"


def small_nets(config, levels=(0, 1, 2)):
    torch.manual_seed(0)
    return {
        level: NeuralQ(level, config.n_side, config.landmarks, config.social_goals, config.params, width=16, depth=2)
        for level in levels
    }


@pytest.fixture
def config():
    config = Config.load(CONFIG)
    config.steps = 2
    return config


def test_policies_are_distributions(config):
    models = NeuralModels(small_nets(config), config)
    s = config.initial_state
    for policy in [*models.l0_policies(1).values(), models.l1_policy(1, config.landmarks[1], -0.5, [(config.landmarks[0], 1.0)])]:
        probs = policy.probs_at(models.grid, s)
        assert len(probs) == 5 and all(p >= 0 for p in probs) and math.isclose(sum(probs), 1, rel_tol=1e-5)


def test_mismatched_config_is_rejected(config):
    nets = small_nets(config)
    config.landmarks = [(1, 1), (0, 3)]
    with pytest.raises(ValueError, match="landmarks"):
        NeuralModels(nets, config)


def test_episode_runs_without_solving(config, tmp_path):
    for level, net in small_nets(config).items():
        net.save(tmp_path / f"q{level}.pt")
    record = run_episode(config, neural=tmp_path, log=lambda *_: None)
    assert record["total_solves"] == 0
    assert len(record["steps"]) == config.steps + 1
    yellow = record["steps"][-1]["beliefs"][0]  # level 2: joint belief over red's (goal, chi)
    assert math.isclose(sum(yellow["other_chi"].values()), 1, rel_tol=1e-6)
    assert math.isclose(sum(yellow["other_goal"].values()), 1, rel_tol=1e-6)
