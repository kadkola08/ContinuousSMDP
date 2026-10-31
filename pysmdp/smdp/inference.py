"""Inferring another agent's goals from its moves (Eq. 5 and 6 of the paper).

The observer keeps a belief over hypotheses about agent j, each with a policy predicting j's
moves under that hypothesis. Each time j moves from s to s', Bayes' rule gives

    P(h | s^{1:t+1}) ∝ P(h | s^{1:t}) * psi_h(s -> s')

where psi_h(s -> s') is the probability that j's policy under hypothesis h leads from s to s'.
The observer sees states, not actions, so every action with the same outcome counts: a blocked
move and staying both explain "j did not move".

Physical goal inference (Eq. 6): hypotheses are j's candidate goals, with j modeled as a
level-0 agent.

Social goal inference (Eq. 5): hypotheses are (physical goal, chi) pairs, with j modeled as a
level-1 agent. Eq. 5 sums over j's physical goal, which is the marginal over chi of this joint
belief; Eq. 6 at level 2 is the marginal over goals.
"""

from collections import defaultdict

from smdp.grid import N_ACTIONS, Grid, State
from smdp.policy import Policy

# The social goals considered in the paper's experiments.
CHIS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def transition_likelihood(grid: Grid, policy: Policy, before: State, after: State) -> float:
    """Probability that ``policy.agent`` moves from ``before`` to ``after``."""
    probs = policy.probs_at(grid, before)
    return sum(
        probs[a] for a in range(N_ACTIONS) if grid.step(before, policy.agent, a) == after
    )


class GoalEstimator:
    """A belief over hypotheses about agent j, updated from j's observed moves."""

    def __init__(self, grid: Grid, policies: dict, prior: dict | None = None):
        """``policies`` maps each hypothesis to j's policy under it. The prior is uniform
        unless given."""
        self.grid = grid
        self.policies = policies
        self.probs = dict(prior) if prior else {h: 1 / len(policies) for h in policies}

    def update(self, before: State, after: State, policies: dict | None = None) -> None:
        """Bayes update for j moving from ``before`` to ``after``.

        ``policies`` replaces the per-hypothesis policies from now on, for when j's policies
        change over time (a level-1 j re-plans as its own beliefs change).
        """
        if policies is not None:
            self.policies = policies
        posterior = {
            h: self.probs[h] * transition_likelihood(self.grid, self.policies[h], before, after)
            for h in self.probs
        }
        total = sum(posterior.values())
        if total <= 1e-30:
            # No hypothesis explains the move (only possible through float underflow at a
            # very high tau). Keep the current belief rather than dividing by zero.
            return
        self.probs = {h: p / total for h, p in posterior.items()}

    def marginal(self, position: int) -> dict:
        """For tuple hypotheses: the belief over the element at ``position``."""
        result = defaultdict(float)
        for h, p in self.probs.items():
            result[h[position]] += p
        return dict(result)

    def belief(self) -> list[tuple]:
        """The belief as (hypothesis, probability) pairs, the form the solver takes."""
        return list(self.probs.items())

    def predicted_policy(self) -> Policy:
        """j's moves predicted by mixing its per-hypothesis policies with the belief (Eq. 4)."""
        return Policy.mixture([(p, self.policies[h]) for h, p in self.probs.items() if p > 0])
