"""Value iteration over the full joint state space.

Within a time step, agent p moves first and the other agent j second. With R(s) the
state-based reward and psi_j the predicted policy of j,

    Q(s, a) = R(s) - c(a) + gamma * sum_b psi_j(s', b) V(step(s', j, b)),   s' = step(s, p, a)
    V(s)    = max_a Q(s, a)

Level 0 (a plain MDP): R is p's physical reward and j is assumed to move uniformly at random.

Level 1 (Eq. 2-4 of the paper): p also has a social goal chi toward j,

    R(s) = r_p(s, g_p) + chi * sum_g P(g) r_j(s, g)

where P is p's belief over j's physical goal, and psi_j is j's level-0 softmax policy, mixed
over the same belief. chi > 0 is helping, chi < 0 hindering, and chi = 0 ignores j's reward.

Level 2 (Eq. 2 at l = 2): p models j as a level-1 agent with its own social goal chi_j toward
p. With P now a joint belief over j's (goal, chi_j), and r^_p the reward j believes p has,

    R(s) = r_p(s, g_p) + chi * sum_{g, chi_j} P(g, chi_j) (r_j(s, g) + chi_j * r^_p(s))

and psi_j is j's level-1 softmax policy, mixed over the joint belief. Even with chi = 0 this
differs from level 0 and 1: p predicts j's moves knowing that j may help or hinder it.

The expectation over j's move depends only on the intermediate state s', so it is computed
once for every state and then looked up per action.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from smdp.grid import AGENTS, N_ACTIONS, Grid, State, other_agent
from smdp.mdp import Params, reward_batch
from smdp.policy import Policy

# A belief over the other agent's physical goal: (goal cell, probability) pairs.
GoalBelief = Sequence[tuple[tuple[int, int], float]]


class Transitions:
    """Next-state index for every (agent, action, state), and the mask of valid states.

    Holds 10 index arrays of n_states each, so it is only meant for small grids.
    """

    def __init__(self, grid: Grid, device=None):
        self.grid = grid
        self.states = grid.decode_range(0, grid.n_states, device)
        self.valid = grid.is_valid_batch(self.states)
        self.next = torch.stack(
            [
                torch.stack(
                    [grid.encode_batch(grid.step_batch(self.states, p, a)) for a in range(N_ACTIONS)]
                )
                for p in AGENTS
            ]
        )  # (agent, action, state)


@dataclass
class Solution:
    values: torch.Tensor  # V(s) for every state index; 0 for invalid states
    q: torch.Tensor  # Q(s, a) from the final values, (n_actions, n_states)
    iterations: int
    residual: float  # max |V_new - V_old| in the last sweep


def q_values(values, rewards, T: Transitions, p: int, params: Params, other_probs=None):
    """Q(s, a) for every action and state, as a (n_actions, n_states) tensor.

    ``other_probs`` is the other agent's policy, (n_actions, n_states); None means uniform.
    """
    other = other_agent(p)
    after_moves = torch.stack([values[T.next[other, b]] for b in range(N_ACTIONS)])
    if other_probs is None:
        after_other = after_moves.mean(dim=0)
    else:
        after_other = (other_probs * after_moves).sum(dim=0)
    future = torch.stack([after_other[T.next[p, a]] for a in range(N_ACTIONS)])
    costs = params.action_costs(values.device)[:, None]
    return rewards[None, :] - costs + params.gamma * future


def value_iterate(
    T: Transitions,
    p: int,
    rewards,
    params: Params,
    other_probs=None,
    tol=1e-4,
    max_iters=2000,
    init=None,
    check_every=10,
) -> Solution:
    """``init`` starts from given values instead of zeros, e.g. the previous solution of a
    slightly different problem, which converges in far fewer sweeps.

    Convergence is checked every ``check_every`` sweeps (and after the last one): each check
    makes the CPU wait for the GPU, so checking every sweep is slow on a GPU.
    """
    rewards = rewards.clone()
    rewards[~T.valid] = 0
    if init is not None:
        values = init.clone()
    else:
        values = torch.zeros(T.grid.n_states, device=T.states.device)
    residual = float("inf")
    for i in range(1, max_iters + 1):
        new = q_values(values, rewards, T, p, params, other_probs).max(dim=0).values
        new[~T.valid] = 0
        if i % check_every == 0 or i == max_iters:
            residual = (new - values).abs().max().item()
        values = new
        if residual < tol:
            break
    q = q_values(values, rewards, T, p, params, other_probs)
    return Solution(values, q, i, residual)


def value_iterate_l0(T: Transitions, p: int, goal, params: Params, **kwargs) -> Solution:
    return value_iterate(T, p, reward_batch(T.states, p, goal, params), params, **kwargs)


def social_rewards(T: Transitions, p: int, goal, chi: float, other_goals: GoalBelief, params):
    """r_p(s, g_p) + chi * E_g[r_j(s, g)] for every state (Eq. 2 at level 1)."""
    j = other_agent(p)
    expected_other = sum(prob * reward_batch(T.states, j, g, params) for g, prob in other_goals)
    return reward_batch(T.states, p, goal, params) + chi * expected_other


def value_iterate_l1(
    T: Transitions,
    p: int,
    goal,
    chi: float,
    other_goals: GoalBelief,
    other_policy: Policy,
    params: Params,
    **kwargs,
) -> Solution:
    """Level 1 for agent p. ``other_policy`` predicts the other agent's moves; build it with
    ``level0_policy_mixture`` from the same belief as ``other_goals``."""
    rewards = social_rewards(T, p, goal, chi, other_goals, params)
    return value_iterate(T, p, rewards, params, other_policy.probs, **kwargs)


def level2_rewards(
    T: Transitions, p: int, goal, chi: float, other_belief, belief_about_me: GoalBelief, params
):
    """Eq. 2 at level 2. ``other_belief`` holds ((g_j, chi_j), prob) pairs for the other agent
    j; ``belief_about_me`` is j's belief over p's goal, which defines r^_p."""
    j = other_agent(p)
    my_reward_seen_by_j = sum(prob * reward_batch(T.states, p, g, params) for g, prob in belief_about_me)
    expected_j = sum(
        prob * (reward_batch(T.states, j, g, params) + chi_j * my_reward_seen_by_j)
        for (g, chi_j), prob in other_belief
    )
    return reward_batch(T.states, p, goal, params) + chi * expected_j


def value_iterate_l2(
    T: Transitions,
    p: int,
    goal,
    chi: float,
    other_belief,
    belief_about_me: GoalBelief,
    other_policy: Policy,
    params: Params,
    **kwargs,
) -> Solution:
    """Level 2 for agent p. ``other_policy`` predicts j's moves; build it by mixing j's
    level-1 policies (``level1_policies``) with ``other_belief``."""
    rewards = level2_rewards(T, p, goal, chi, other_belief, belief_about_me, params)
    return value_iterate(T, p, rewards, params, other_policy.probs, **kwargs)


def level0_policies(T: Transitions, j: int, goals, params: Params) -> dict:
    """j's level-0 softmax policy for each candidate goal, as {goal: Policy}."""
    return {g: Policy.from_q(j, value_iterate_l0(T, j, g, params).q, params.tau) for g in goals}


def level0_policy_mixture(T: Transitions, j: int, other_goals: GoalBelief, params: Params) -> Policy:
    """j's predicted policy: its level-0 softmax policy per goal, mixed by the belief (Eq. 4)."""
    likely = [g for g, prob in other_goals if prob > 0]
    policies = level0_policies(T, j, likely, params)
    return Policy.mixture([(prob, policies[g]) for g, prob in other_goals if prob > 0])


def level1_policies(
    T: Transitions, j: int, goals, chis, other_goals: GoalBelief, other_policy: Policy, params
) -> dict:
    """j's level-1 softmax policy for each (physical goal, chi) hypothesis.

    ``other_goals`` and ``other_policy`` are j's belief about the other agent's goal and its
    prediction of the other agent's moves, shared by all hypotheses.
    """
    return {
        (g, chi): Policy.from_q(
            j, value_iterate_l1(T, j, g, chi, other_goals, other_policy, params).q, params.tau
        )
        for g in goals
        for chi in chis
    }


def q_at(T: Transitions, values, s: State, p: int, rewards, params: Params, other_probs=None):
    """Q(s, a) for one state with the scalar step function. A reference for tests."""
    grid, other = T.grid, other_agent(p)
    q = []
    for a in range(N_ACTIONS):
        s1 = grid.step(s, p, a)
        i1 = grid.encode(s1)
        future = 0.0
        for b in range(N_ACTIONS):
            weight = 1 / N_ACTIONS if other_probs is None else other_probs[b, i1].item()
            future += weight * values[grid.encode(grid.step(s1, other, b))].item()
        q.append(rewards[grid.encode(s)].item() - params.action_cost(a) + params.gamma * future)
    return q
