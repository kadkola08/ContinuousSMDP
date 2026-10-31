"""Physical reward and planning parameters.

Each agent's physical goal is a landmark cell; the agent wants its own can on that cell.
Following the paper (Section 4), the reward is

    r(s, g) = max(rho * (1 - distance(can, g) / delta), 0)

with the Euclidean distance from the agent's can to the goal.
"""

import math
from dataclasses import dataclass

import torch

from smdp.grid import ACTIONS, N_ACTIONS, State, can_of, position


@dataclass(frozen=True)
class Params:
    gamma: float = 0.95
    rho: float = 1.0  # reward with the can on the goal
    delta: float = 5.0  # distance at which the reward reaches 0
    move_cost: float = 0.05
    stay_cost: float = 0.0
    tau: float = 5.0  # softmax temperature: how rational agents assume each other to be

    def action_cost(self, action: int) -> float:
        return self.stay_cost if ACTIONS[action] == (0, 0) else self.move_cost

    def action_costs(self, device=None) -> torch.Tensor:
        return torch.tensor([self.action_cost(a) for a in range(N_ACTIONS)], device=device)


def reward(s: State, p: int, goal: tuple[int, int], params: Params) -> float:
    row, col = position(s, can_of(p))
    distance = math.hypot(goal[0] - row, goal[1] - col)
    return max(params.rho * (1 - distance / params.delta), 0.0)


def reward_batch(states: torch.Tensor, p: int, goal: tuple[int, int], params: Params):
    can = states[:, 2 * can_of(p) : 2 * can_of(p) + 2].float()
    target = torch.tensor(goal, dtype=torch.float32, device=states.device)
    distance = (can - target).norm(dim=1)
    return (params.rho * (1 - distance / params.delta)).clamp(min=0)
