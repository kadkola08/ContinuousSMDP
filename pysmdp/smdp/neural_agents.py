"""Agents that plan with the Q-networks of smdp.neural instead of value iteration.

The beliefs and goal inference are those of smdp.agents, unchanged. Wherever those agents
solve an MDP, these evaluate a network at the state they need instead:

    own Q       Q_l(s, .; own goal, chi, current beliefs), one forward pass at s
    hypotheses  the other agent's policy under each hypothesis is softmax(tau * Q) of the
                network one level down (Eq. 7), evaluated only at the states it is asked about

Nothing is solved, so there is no caching, no belief rounding and no table over the state
space: an agent can re-plan for its exact beliefs at every step.
"""

from pathlib import Path

import torch

from smdp.agents import Level0Agent, Level1Agent, Level2Agent
from smdp.config import Config
from smdp.grid import Grid, State
from smdp.neural import Hypothesis, NeuralQ


class NetPolicy:
    """Agent ``agent``'s softmax policy from a network under one hypothesis.

    Provides the part of smdp.policy.Policy that goal inference uses.
    """

    def __init__(self, agent: int, model: NeuralQ, hypothesis: Hypothesis):
        self.agent, self.model, self.hypothesis = agent, model, hypothesis

    def probs_at(self, grid: Grid, s: State) -> list[float]:
        q = self.model.q(torch.tensor([s], device=self.model.landmarks.device), self.agent, self.hypothesis)
        return torch.softmax(self.model.params.tau * q, dim=1)[0].tolist()


class NeuralModels:
    """The trained network for each level, in place of smdp.agents.Models."""

    def __init__(self, nets: dict, config: Config):
        for level, net in nets.items():
            mismatch = [
                name for name, ours, theirs in (
                    ("n_side", config.n_side, net.n_side),
                    ("landmarks", config.landmarks, net.landmarks_list),
                    ("social_goals", config.social_goals, net.chis_list),
                    ("params", config.params, net.params),
                ) if ours != theirs
            ]
            if mismatch:
                raise ValueError(f"level-{level} network was trained with different {', '.join(mismatch)}")
        self.nets = nets
        self.grid = config.grid
        self.params = config.params
        self.landmarks = config.landmarks
        self.chis = config.social_goals
        self.device = next(iter(nets.values())).landmarks.device
        self.solves = 0  # nothing is ever solved; kept for the episode's progress reports

    @classmethod
    def load(cls, directory, config: Config, device=None) -> "NeuralModels":
        """Loads q0.pt up to the highest level any agent in ``config`` needs."""
        top = max(a.level for a in config.agents)
        return cls({level: NeuralQ.load(Path(directory) / f"q{level}.pt", device) for level in range(top + 1)}, config)

    def hypothesis(self, **fields) -> Hypothesis:
        """A one-row Hypothesis from plain values."""
        return Hypothesis.constant(1, self.device, **fields)

    def goal_vector(self, belief) -> list[float]:
        """A belief as (goal, prob) pairs, as a vector over the landmarks."""
        probs = dict(belief)
        return [probs.get(g, 0.0) for g in self.landmarks]

    def joint_vector(self, probs: dict) -> list[float]:
        """A belief over (goal, chi) as a vector over landmarks x social goals."""
        return [probs.get((g, c), 0.0) for g in self.landmarks for c in self.chis]

    def q(self, level: int, s: State, p: int, **fields) -> list[float]:
        states = torch.tensor([s], device=self.device)
        with torch.no_grad():
            return self.nets[level].q(states, p, self.hypothesis(**fields))[0].tolist()

    def l0_policy(self, p: int, goal) -> NetPolicy:
        return NetPolicy(p, self.nets[0], self.hypothesis(goal=goal))

    def l0_policies(self, p: int) -> dict:
        return {g: self.l0_policy(p, g) for g in self.landmarks}

    def l1_policy(self, j: int, goal, chi: float, belief_about_other) -> NetPolicy:
        """j's level-1 policy for (goal, chi) given j's belief over the other agent's goal."""
        return NetPolicy(j, self.nets[1], self.hypothesis(goal=goal, chi=chi, belief=self.goal_vector(belief_about_other)))


class NeuralLevel0Agent(Level0Agent):
    def plan(self):
        pass

    def q_at(self, s):
        return self.models.q(0, s, self.p, goal=self.goal)


class NeuralLevel1Agent(Level1Agent):
    def plan(self):
        pass

    def q_at(self, s):
        belief = self.models.goal_vector(self.about_other.belief())
        return self.models.q(1, s, self.p, goal=self.goal, chi=self.chi, belief=belief)


class NeuralLevel2Agent(Level2Agent):
    def plan(self):
        pass

    def q_at(self, s):
        models = self.models
        return models.q(
            2, s, self.p, goal=self.goal, chi=self.chi,
            joint=models.joint_vector(self.about_other.probs),
            about_me=models.goal_vector(self.about_me.belief()),
        )


NEURAL_AGENT_CLASSES = {0: NeuralLevel0Agent, 1: NeuralLevel1Agent, 2: NeuralLevel2Agent}


def make_neural_agents(config: Config, models: NeuralModels, seed: int = 0) -> list:
    return [
        NEURAL_AGENT_CLASSES[a.level](p, config, models, seed) for p, a in enumerate(config.agents)
    ]
