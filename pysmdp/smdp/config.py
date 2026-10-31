"""Experiment configuration, read from JSON.

Example (positions are [row, col]):

    {
      "name": "blocking_help",
      "n_side": 5,
      "landmarks": [[0, 1], [0, 3]],
      "agents": [
        {"goal": 0, "level": 2, "chi": 0},
        {"goal": 1, "level": 1, "chi": 1}
      ],
      "initial_state": {"agents": [[4, 4], [3, 1]], "cans": [[2, 1], [0, 3]]},
      "steps": 10,
      "social_goals": [-2, -1, -0.5, 0, 0.5, 1, 2],
      "params": {"gamma": 0.95, "tau": 5}
    }

"goal" indexes into "landmarks"; every landmark is a candidate goal when agents infer each
other's goals. "social_goals" are the chi hypotheses a level-2 agent considers. "params" may
override any field of smdp.mdp.Params. Optional "tol" (default 1e-3) and "belief_resolution"
(default 0.05) trade accuracy for speed; see Config. Agent 0 is drawn yellow and agent 1 red.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from smdp.grid import Grid, State, make_state
from smdp.inference import CHIS
from smdp.mdp import Params


@dataclass
class AgentConfig:
    goal: int  # index into Config.landmarks
    level: int  # 0, 1 or 2
    chi: float = 0.0  # social goal toward the other agent (levels 1 and 2)


@dataclass
class Config:
    name: str
    n_side: int
    landmarks: list[tuple[int, int]]
    agents: list[AgentConfig]
    initial_state: State
    steps: int = 10
    social_goals: list[float] = field(default_factory=lambda: list(CHIS))
    params: Params = field(default_factory=Params)
    tol: float = 1e-3  # value-iteration convergence threshold
    # A level-2 agent's hypotheses about the other agent depend on what the other agent
    # believes about it. That belief is rounded to this step before solving, so the
    # hypotheses are only re-solved when it changes noticeably.
    belief_resolution: float = 0.05

    @property
    def grid(self) -> Grid:
        return Grid(self.n_side)

    def goal_of(self, p: int) -> tuple[int, int]:
        return self.landmarks[self.agents[p].goal]

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(Params)}
        unknown = set(data.get("params", {})) - known
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}")
        state = data["initial_state"]
        config = cls(
            name=data["name"],
            n_side=data["n_side"],
            landmarks=[tuple(cell) for cell in data["landmarks"]],
            agents=[
                AgentConfig(goal=a["goal"], level=a["level"], chi=float(a.get("chi", 0.0)))
                for a in data["agents"]
            ],
            initial_state=make_state(*map(tuple, state["agents"]), *map(tuple, state["cans"])),
            steps=data.get("steps", 10),
            social_goals=[float(c) for c in data.get("social_goals", CHIS)],
            params=Params(**data.get("params", {})),
            tol=data.get("tol", 1e-3),
            belief_resolution=data.get("belief_resolution", 0.05),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self):
        if len(self.agents) != 2:
            raise ValueError("exactly two agents are supported")
        for p, agent in enumerate(self.agents):
            if agent.level not in (0, 1, 2):
                raise ValueError(f"agent {p}: level must be 0, 1 or 2, got {agent.level}")
            if not 0 <= agent.goal < len(self.landmarks):
                raise ValueError(f"agent {p}: goal {agent.goal} is not a landmark index")
        grid = self.grid
        if not all(grid.in_bounds(r, c) for r, c in zip(self.initial_state[::2], self.initial_state[1::2])):
            raise ValueError("initial_state has a position outside the grid")
        if not grid.is_valid(self.initial_state):
            raise ValueError("initial_state puts two entities on the same cell")
