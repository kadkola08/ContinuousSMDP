"""Agents at levels 0, 1 and 2 that infer each other's goals while acting (Algorithm 1).

Each agent plans before its own move, using everything it has observed so far, and acts
greedily on its Q-values. Agent p's "other" is j = 1 - p.

Level 0  plans once for its own goal, treating j as random.
Level 1  keeps a belief over j's physical goal (Eq. 6), modelling j as level 0, and re-plans
         a level-1 MDP with it (Eq. 2-4).
Level 2  keeps a joint belief over j's (physical goal, chi) (Eq. 5), modelling j as level 1,
         and re-plans a level-2 MDP. j's level-1 policies depend on what j believes about p's
         goal, so p also tracks that belief by running j's goal inference on its own moves.

Models caches every solved policy, keyed by the belief it was solved for, and warm starts each
solve from the previous solution for the same agent, goal and chi. The belief behind a level-2
agent's hypotheses is rounded to Config.belief_resolution first, so those 2 x len(chis)
policies are only re-solved when it changes noticeably.
"""

import random

import torch

from smdp.config import Config
from smdp.grid import N_ACTIONS, State, other_agent
from smdp.inference import GoalEstimator
from smdp.policy import Policy
from smdp.solver import (
    Transitions,
    value_iterate_l0,
    value_iterate_l1,
    value_iterate_l2,
)

BELIEF_DECIMALS = 3


def belief_key(belief) -> tuple:
    return tuple((h, round(p, BELIEF_DECIMALS)) for h, p in belief)


def quantize(belief, resolution: float) -> list:
    """Rounds each probability to a multiple of ``resolution`` and renormalizes."""
    rounded = [(h, round(p / resolution) * resolution) for h, p in belief]
    total = sum(p for _, p in rounded)
    return [(h, p / total) for h, p in rounded]


class Models:
    """Solved policies shared by both agents, so nothing is solved twice."""

    def __init__(self, T: Transitions, config: Config):
        self.T = T
        self.grid = T.grid
        self.params = config.params
        self.landmarks = config.landmarks
        self.tol = config.tol
        self.resolution = config.belief_resolution
        self._l0 = {}
        self._l1 = {}
        self._warm = {}
        self.solves = 0  # number of value iterations run, for progress reports

    def l0_solution(self, p: int, goal):
        if (p, goal) not in self._l0:
            self._l0[(p, goal)] = value_iterate_l0(self.T, p, goal, self.params, tol=self.tol)
            self.solves += 1
        return self._l0[(p, goal)]

    def l0_policy(self, p: int, goal) -> Policy:
        return Policy.from_q(p, self.l0_solution(p, goal).q, self.params.tau)

    def l0_policies(self, p: int) -> dict:
        return {g: self.l0_policy(p, g) for g in self.landmarks}

    def l1_policy(self, j: int, goal, chi: float, belief_about_other) -> Policy:
        """j's level-1 policy for (goal, chi) given j's belief over the other agent's goal."""
        belief_about_other = quantize(belief_about_other, self.resolution)
        key = (j, goal, chi, belief_key(belief_about_other))
        if key not in self._l1:
            other = other_agent(j)
            other_model = Policy.mixture(
                [(p, self.l0_policy(other, g)) for g, p in belief_about_other if p > 0]
            )
            warm = self._warm.get(("l1", j, goal, chi))
            solution = value_iterate_l1(
                self.T, j, goal, chi, belief_about_other, other_model, self.params,
                init=warm, tol=self.tol,
            )
            self._warm[("l1", j, goal, chi)] = solution.values
            self._l1[key] = Policy.from_q(j, solution.q, self.params.tau)
            self.solves += 1
        return self._l1[key]

    def warm(self, key):
        return self._warm.get(key)

    def store_warm(self, key, values):
        self._warm[key] = values


class Agent:
    level = 0

    def __init__(self, p: int, config: Config, models: Models, seed: int = 0):
        self.p = p
        self.j = other_agent(p)
        self.goal = config.goal_of(p)
        self.chi = config.agents[p].chi
        self.models = models
        self.grid = models.grid
        self.q = None
        self.rng = random.Random(seed + p)

    def plan(self) -> None:
        raise NotImplementedError

    def q_at(self, s: State) -> list[float]:
        """Q(s, a) for every action, from the last plan."""
        return self.q[:, self.grid.encode(s)].tolist()

    def act(self, s: State) -> int:
        """The greedy action, ties broken at random."""
        q = self.q_at(s)
        best = max(q)
        return self.rng.choice([a for a in range(N_ACTIONS) if best - q[a] < 1e-6])

    def action_probs(self, s: State) -> list[float]:
        """This agent's own softmax policy at s, for the record."""
        return torch.softmax(self.models.params.tau * torch.tensor(self.q_at(s)), dim=0).tolist()

    def observe_own(self, before: State, after: State) -> None:
        pass

    def observe_other(self, before: State, after: State) -> None:
        pass

    def beliefs(self) -> dict:
        return {}


class Level0Agent(Agent):
    level = 0

    def plan(self):
        if self.q is None:
            self.q = self.models.l0_solution(self.p, self.goal).q


class Level1Agent(Agent):
    level = 1

    def __init__(self, p, config, models, seed=0):
        super().__init__(p, config, models, seed)
        self.about_other = GoalEstimator(self.grid, models.l0_policies(self.j))
        self._planned_for = None

    def observe_other(self, before, after):
        self.about_other.update(before, after)

    def plan(self):
        key = belief_key(self.about_other.belief())
        if key == self._planned_for:
            return
        warm_key = ("own-l1", self.p)
        solution = value_iterate_l1(
            self.models.T,
            self.p,
            self.goal,
            self.chi,
            self.about_other.belief(),
            self.about_other.predicted_policy(),
            self.models.params,
            init=self.models.warm(warm_key),
            tol=self.models.tol,
        )
        self.models.store_warm(warm_key, solution.values)
        self.models.solves += 1
        self.q = solution.q
        self._planned_for = key

    def beliefs(self):
        return {"other_goal": dict(self.about_other.probs)}


class Level2Agent(Agent):
    level = 2

    def __init__(self, p, config, models, seed=0):
        super().__init__(p, config, models, seed)
        self.chis = config.social_goals
        # What j believes about p's goal: j's goal inference, run on p's own moves.
        self.about_me = GoalEstimator(self.grid, models.l0_policies(p))
        self.about_other = GoalEstimator(self.grid, self._hypotheses())
        self._planned_for = None

    def _hypotheses(self) -> dict:
        """j's level-1 policy for every (goal, chi), given j's current belief about p."""
        belief = self.about_me.belief()
        return {
            (g, chi): self.models.l1_policy(self.j, g, chi, belief)
            for g in self.models.landmarks
            for chi in self.chis
        }

    def observe_own(self, before, after):
        self.about_me.update(before, after)

    def observe_other(self, before, after):
        # j moved knowing p's latest move, so use j's policies under its updated belief.
        self.about_other.update(before, after, policies=self._hypotheses())

    def plan(self):
        key = (belief_key(self.about_other.belief()), belief_key(self.about_me.belief()))
        if key == self._planned_for:
            return
        self.about_other.policies = self._hypotheses()
        warm_key = ("own-l2", self.p)
        solution = value_iterate_l2(
            self.models.T,
            self.p,
            self.goal,
            self.chi,
            self.about_other.belief(),
            self.about_me.belief(),
            self.about_other.predicted_policy(),
            self.models.params,
            init=self.models.warm(warm_key),
            tol=self.models.tol,
        )
        self.models.store_warm(warm_key, solution.values)
        self.models.solves += 1
        self.q = solution.q
        self._planned_for = key

    def beliefs(self):
        social = self.about_other.marginal(1)
        return {
            "other_goal": self.about_other.marginal(0),
            "other_chi": social,
            "other_expected_chi": sum(c * p for c, p in social.items()),
            "about_me": dict(self.about_me.probs),
        }


AGENT_CLASSES = {0: Level0Agent, 1: Level1Agent, 2: Level2Agent}


def make_agents(config: Config, models: Models, seed: int = 0) -> list[Agent]:
    return [
        AGENT_CLASSES[a.level](p, config, models, seed) for p, a in enumerate(config.agents)
    ]
