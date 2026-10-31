"""Softmax policies (Eq. 7 of the paper).

An agent predicts another agent's moves with

    psi(s, a) = exp(tau * Q(s, a)) / sum_b exp(tau * Q(s, b))

tau sets how rational the other agent is assumed to be: tau = 0 is uniformly random, and as
tau grows the policy approaches always taking the best action. Only differences between the
Q-values of one state matter, so their overall scale (around 15-20 here) has no effect.
"""

import random
from dataclasses import dataclass

import torch

from smdp.grid import N_ACTIONS, Grid, State


@dataclass
class Policy:
    """Action probabilities of agent ``agent`` for every state, shape (n_actions, n_states).

    Only meaningful for valid states.
    """

    agent: int
    probs: torch.Tensor

    @classmethod
    def from_q(cls, agent: int, q: torch.Tensor, tau: float) -> "Policy":
        return cls(agent, torch.softmax(tau * q, dim=0))

    @classmethod
    def mixture(cls, weighted: list[tuple[float, "Policy"]]) -> "Policy":
        """sum_k w_k psi_k for policies of the same agent; the weights must sum to 1."""
        agent = weighted[0][1].agent
        assert all(policy.agent == agent for _, policy in weighted)
        return cls(agent, sum(w * policy.probs for w, policy in weighted))

    def probs_at(self, grid: Grid, s: State) -> list[float]:
        return self.probs[:, grid.encode(s)].tolist()

    def prob(self, grid: Grid, s: State, action: int) -> float:
        return self.probs[action, grid.encode(s)].item()

    def sample(self, grid: Grid, s: State, rng: random.Random) -> int:
        return rng.choices(range(N_ACTIONS), weights=self.probs_at(grid, s))[0]
