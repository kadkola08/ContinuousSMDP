"""Neural Q-functions for Social MDPs, trained by fitted value iteration.

The tabular solver (smdp.solver) stores Q(s, a) for every state, once per hypothesis. Here Q is
a network that also takes the hypothesis as input, so one network covers every goal, social
goal and belief, and the recursion between levels becomes forward passes instead of solves:

    level 0  Q0(s, a; g)
    level 1  Q1(s, a; g, chi, b)          b: belief over the other agent's goal
    level 2  Q2(s, a; g, chi, B, b_me)    B: joint belief over the other's (goal, chi_j);
                                          b_me: what the other agent believes about my goal

Networks see states from the acting agent's point of view (``egocentric``). The dynamics and
rewards are symmetric under swapping the two agents and their cans, so one network per level
serves both agents.

``fit`` trains a network with the backup of smdp.solver.q_values, on sampled states:

    target(s, a) = R(s) - c(a) + gamma * sum_b psi_j(s', b) max_a' Q_target(step(s', j, b), a')

where s' = step(s, me, a), R is the level's reward (Eq. 2, closed form given the hypothesis)
and psi_j is the other agent's softmax policy from the frozen network one level down (Eq. 7),
mixed over the belief. Only ``sample_states`` knows the state space is a grid, so the rest
applies to any batched simulator.
"""

import copy
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn

from smdp.grid import N_ACTIONS, N_ENTITIES, Grid, can_of
from smdp.mdp import Params
from smdp.solver import Solution

# Swaps agent 0 with agent 1 and can 0 with can 1.
SWAP = [2, 3, 0, 1, 6, 7, 4, 5]


def egocentric(states: torch.Tensor, p: int) -> torch.Tensor:
    """States as agent p sees them: itself as agent 0 and its can as can 0."""
    return states if p == 0 else states[:, SWAP]


def reward_rows(states: torch.Tensor, p: int, goals: torch.Tensor, params: Params):
    """r_p(s, g) for every row, with goals (N, 2) or a single (2,) goal."""
    can = states[:, 2 * can_of(p) : 2 * can_of(p) + 2].float()
    distance = (can - goals).norm(dim=1)
    return (params.rho * (1 - distance / params.delta)).clamp(min=0)


@dataclass
class Hypothesis:
    """One hypothesis per row: what the acting agent wants and believes.

    ``goal`` is a cell, ``belief`` and ``about_me`` are over Config.landmarks, and ``joint``
    is over (landmark k, chi m) flattened as k * len(chis) + m.
    """

    goal: torch.Tensor  # (N, 2)
    chi: torch.Tensor | None = None  # (N,), levels 1 and 2
    belief: torch.Tensor | None = None  # (N, K), level 1
    joint: torch.Tensor | None = None  # (N, K * M), level 2
    about_me: torch.Tensor | None = None  # (N, K), level 2

    @classmethod
    def constant(cls, n: int, device=None, **fields) -> "Hypothesis":
        """The same hypothesis on n rows, from plain values, e.g. goal=(0, 1), chi=1.0,
        belief=[1.0, 0.0]."""
        tensors = {name: torch.tensor(value, dtype=torch.float32, device=device) for name, value in fields.items()}
        return cls(**{name: t.expand(n, *t.shape) for name, t in tensors.items()})

    def __len__(self):
        return len(self.goal)

    def repeat(self, n: int) -> "Hypothesis":
        """Each row repeated n times in a row, to match states expanded over actions."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return Hypothesis(
            **{k: None if v is None else v.repeat_interleave(n, dim=0) for k, v in values.items()}
        )


class QNet(nn.Module):
    """An MLP with a dueling head: Q = scale * (V + A - mean(A)).

    Softmax policies only depend on differences between actions, which the advantage head
    learns separately from the much larger V. ``scale`` is about the size of the returns.
    """

    def __init__(self, n_in: int, width: int, depth: int, scale: float):
        super().__init__()
        layers, size = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, N_ACTIONS)
        self.scale = scale

    def forward(self, x):
        h = self.trunk(x)
        a = self.advantage(h)
        return self.scale * (self.value(h) + a - a.mean(dim=1, keepdim=True))


class NeuralQ(nn.Module):
    """Q at one level for any hypothesis, from the acting agent's point of view."""

    def __init__(self, level: int, n_side: int, landmarks, chis, params: Params,
                 width: int = 256, depth: int = 4):
        super().__init__()
        self.level, self.n_side, self.params = level, n_side, params
        self.chis_list = [float(c) for c in chis]
        self.landmarks_list = [tuple(g) for g in landmarks]
        self.register_buffer("landmarks", torch.tensor(self.landmarks_list, dtype=torch.float32))
        self.register_buffer("chis", torch.tensor(self.chis_list, dtype=torch.float32))
        self.arch = {"width": width, "depth": depth}
        n_in = self._features(torch.zeros(1, 2 * N_ENTITIES, dtype=torch.long),
                              Hypothesis(**self._dummy())).shape[1]
        self.net = QNet(n_in, width, depth, scale=1 / (1 - params.gamma))

    @property
    def n_landmarks(self) -> int:
        return len(self.landmarks_list)

    def _dummy(self) -> dict:
        k, m = self.n_landmarks, len(self.chis_list)
        h = {"goal": torch.zeros(1, 2)}
        if self.level >= 1:
            h["chi"] = torch.zeros(1)
        if self.level == 1:
            h["belief"] = torch.zeros(1, k)
        if self.level == 2:
            h["joint"], h["about_me"] = torch.zeros(1, k * m), torch.zeros(1, k)
        return h

    def _features(self, states, h: Hypothesis):
        """Positions, their pairwise offsets, and each can's offset to the goal and landmarks,
        all scaled to about [-1, 1], followed by the rest of the hypothesis."""
        scale = 2 / (self.n_side - 1)
        pos = states.view(-1, N_ENTITIES, 2).float() * scale - 1
        i, j = torch.triu_indices(N_ENTITIES, N_ENTITIES, 1)
        pairs = pos[:, i] - pos[:, j]
        goal = h.goal.to(pos) * scale - 1
        to_goal = pos[:, 2] - goal
        to_landmarks = pos[:, 2:, None] - (self.landmarks.to(pos) * scale - 1)
        parts = [
            pos.flatten(1), pairs.flatten(1), goal, to_goal, to_goal.norm(dim=1, keepdim=True),
            to_landmarks.flatten(1), to_landmarks.norm(dim=3).flatten(1),
        ]
        if self.level >= 1:
            parts.append(h.chi.to(pos)[:, None])
        if self.level == 1:
            parts.append(h.belief.to(pos))
        if self.level == 2:
            parts += [h.joint.to(pos), h.about_me.to(pos)]
        return torch.cat(parts, dim=1)

    def forward(self, states, h: Hypothesis):
        """Q(s, .) as (N, n_actions), for states already in this agent's frame."""
        return self.net(self._features(states, h))

    def q(self, states, p: int, h: Hypothesis):
        """Q(s, .) for agent p, with states in the world frame."""
        return self(egocentric(states, p), h)

    # ----- Saving -----

    def save(self, path):
        torch.save(
            {
                "level": self.level, "n_side": self.n_side, "landmarks": self.landmarks_list,
                "chis": self.chis_list, "params": self.params, "arch": self.arch,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None) -> "NeuralQ":
        data = torch.load(path, map_location=device, weights_only=False)
        model = cls(data["level"], data["n_side"], data["landmarks"], data["chis"],
                    data["params"], **data["arch"])
        model.load_state_dict(data["state_dict"])
        return model.to(device)


# ----- The level's reward and the other agent's policy, in the acting agent's frame -----


def level_rewards(model: NeuralQ, states, h: Hypothesis):
    """R(s) at the model's level (Eq. 2); the acting agent is agent 0 of ``states``."""
    params = model.params
    own = reward_rows(states, 0, h.goal.to(states.device), params)
    if model.level == 0:
        return own
    others = torch.stack([reward_rows(states, 1, g, params) for g in model.landmarks], dim=1)
    if model.level == 1:
        return own + h.chi * (h.belief * others).sum(dim=1)
    mine = torch.stack([reward_rows(states, 0, g, params) for g in model.landmarks], dim=1)
    seen_by_other = (h.about_me * mine).sum(dim=1)  # r^_p: my reward as the other agent sees it
    joint = h.joint.view(len(h), model.n_landmarks, -1)
    expected_other = (joint.sum(dim=2) * others).sum(dim=1) + (joint * model.chis).sum(dim=(1, 2)) * seen_by_other
    return own + h.chi * expected_other


def other_policy(model: NeuralQ, lower: NeuralQ | None, states, h: Hypothesis):
    """psi_j(s, .) as (N, n_actions): the other agent (agent 1 of ``states``) under ``lower``,
    one level down, mixed over the acting agent's belief (Eq. 4 and 7)."""
    n, tau = len(states), model.params.tau
    if model.level == 0:
        return torch.full((n, N_ACTIONS), 1 / N_ACTIONS, device=states.device)
    view = egocentric(states, 1)
    probs = torch.zeros(n, N_ACTIONS, device=states.device)
    if model.level == 1:
        for k, g in enumerate(model.landmarks):
            q = lower(view, Hypothesis(goal=g.expand(n, 2)))
            probs += h.belief[:, k, None] * torch.softmax(tau * q, dim=1)
        return probs
    joint = h.joint.view(n, model.n_landmarks, -1)
    for k, g in enumerate(model.landmarks):
        for m, chi in enumerate(model.chis):
            weight = joint[:, k, m]
            if not weight.any():
                continue
            q = lower(view, Hypothesis(goal=g.expand(n, 2), chi=chi.expand(n), belief=h.about_me))
            probs += weight[:, None] * torch.softmax(tau * q, dim=1)
    return probs


def bellman_targets(grid: Grid, states, h: Hypothesis, rewards, params: Params, value_fn, other_fn):
    """The backup above for every action, as (N, n_actions).

    ``value_fn(states, h)`` gives V for each row and ``other_fn(states, h)`` gives psi_j as
    (N, n_actions); both get states in the acting agent's frame.
    """
    n = len(states)
    s1 = torch.stack([grid.step_batch(states, 0, a) for a in range(N_ACTIONS)], dim=1).view(-1, 2 * N_ENTITIES)
    h1 = h.repeat(N_ACTIONS)
    psi = other_fn(s1, h1).view(n, N_ACTIONS, N_ACTIONS)  # [state, my action, their action]
    s2 = torch.stack([grid.step_batch(s1, 1, b) for b in range(N_ACTIONS)], dim=1).view(-1, 2 * N_ENTITIES)
    v2 = value_fn(s2, h1.repeat(N_ACTIONS)).view(n, N_ACTIONS, N_ACTIONS)
    costs = params.action_costs(states.device)
    return rewards[:, None] - costs[None, :] + params.gamma * (psi * v2).sum(dim=2)


# ----- Sampling training data -----


def sample_states(grid: Grid, n: int, device=None):
    """n valid states, uniformly at random."""
    chunks, total = [], 0
    while total < n:
        s = torch.randint(0, grid.n_side, (2 * n, 2 * N_ENTITIES), device=device)
        s = s[grid.is_valid_batch(s)]
        chunks.append(s)
        total += len(s)
    return torch.cat(chunks)[:n]


def sample_simplex(n: int, k: int, device=None):
    """Beliefs over k options: a mix of certain, sharp, flat and random ones."""
    kind = torch.randint(0, 4, (n,), device=device)
    one_hot = torch.nn.functional.one_hot(torch.randint(0, k, (n,), device=device), k).float()
    flat = torch.distributions.Dirichlet(torch.ones(k, device=device)).sample((n,))
    sparse = torch.distributions.Dirichlet(torch.full((k,), 0.3, device=device)).sample((n,))
    eps = torch.rand(n, 1, device=device) * 0.3
    sharp = (1 - eps) * one_hot + eps * flat
    choices = torch.stack([one_hot, sharp, flat, sparse])
    return choices[kind, torch.arange(n, device=device)]


def sample_hypotheses(model: NeuralQ, n: int, device=None, chi_range=(-2.0, 2.0)) -> Hypothesis:
    """Random hypotheses at the model's level. chi is continuous over ``chi_range``."""
    k, m = model.n_landmarks, len(model.chis_list)
    goal = model.landmarks.to(device)[torch.randint(0, k, (n,), device=device)]
    if model.level == 0:
        return Hypothesis(goal=goal)
    chi = torch.empty(n, device=device).uniform_(*chi_range)
    if model.level == 1:
        return Hypothesis(goal=goal, chi=chi, belief=sample_simplex(n, k, device))
    return Hypothesis(goal=goal, chi=chi, joint=sample_simplex(n, k * m, device),
                      about_me=sample_simplex(n, k, device))


# ----- Training -----


def fit(model: NeuralQ, lower: NeuralQ | None, grid: Grid, steps=20000, batch=1024, lr=1e-3,
        polyak=0.01, log_every=1000, evaluate=None, log=print):
    """Fitted value iteration for ``model``, with ``lower`` frozen as the other agent's model.

    A slowly updated copy of the network gives the bootstrapped values, as in DQN.
    ``evaluate()``, if given, is called every ``log_every`` steps and its result is logged.
    """
    device = model.landmarks.device
    target = copy.deepcopy(model).requires_grad_(False)
    if lower is not None:
        lower.requires_grad_(False).eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    scale = model.net.scale
    running = 0.0
    for step in range(1, steps + 1):
        states = sample_states(grid, batch, device)
        h = sample_hypotheses(model, batch, device)
        with torch.no_grad():
            y = bellman_targets(
                grid, states, h, level_rewards(model, states, h), model.params,
                value_fn=lambda s, hh: target(s, hh).max(dim=1).values,
                other_fn=lambda s, hh: other_policy(model, lower, s, hh),
            )
        loss = torch.nn.functional.mse_loss(model(states, h) / scale, y / scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for p, tp in zip(model.parameters(), target.parameters()):
                tp.lerp_(p, polyak)
        running += loss.item() if step % 50 == 0 else 0.0
        if step % log_every == 0 or step == steps:
            message = f"  step {step:6d}  loss {running / max(1, log_every // 50):.2e}"
            if evaluate is not None:
                message += "  " + evaluate()
            log(message)
            running = 0.0
    return model


# ----- Whole-grid tables, for figures and comparisons with smdp.solver -----


def load_levels(directory, levels, params: Params, landmarks, device=None) -> dict:
    """{level: network} from q<level>.pt in ``directory``, checked against the caller's setup."""
    nets = {}
    for level in levels:
        net = NeuralQ.load(Path(directory) / f"q{level}.pt", device)
        if net.params != params:
            raise ValueError(f"q{level}.pt was trained with {net.params}, not {params}")
        if net.landmarks_list != [tuple(g) for g in landmarks]:
            raise ValueError(f"q{level}.pt was trained with landmarks {net.landmarks_list}, not {list(landmarks)}")
        nets[level] = net.eval()
    return nets


def q_table(model: NeuralQ, grid: Grid, p: int, chunk: int = 65536, **fields) -> torch.Tensor:
    """Agent p's Q under one hypothesis on every state, as (n_actions, n_states), the layout of
    smdp.solver.Solution.q. Invalid states get 0. ``fields`` are as in Hypothesis.constant."""
    device = model.landmarks.device
    table = torch.zeros(N_ACTIONS, grid.n_states, device=device)
    with torch.no_grad():
        for start in range(0, grid.n_states, chunk):
            index = torch.arange(start, min(start + chunk, grid.n_states), device=device)
            states = grid.decode_range(start, start + len(index), device)
            valid = grid.is_valid_batch(states)
            h = Hypothesis.constant(int(valid.sum()), device, **fields)
            table[:, index[valid]] = model.q(states[valid], p, h).T
    return table


def as_solution(q: torch.Tensor) -> Solution:
    """A Q table as a smdp.solver.Solution, for code written against the tabular solver."""
    return Solution(values=q.max(dim=0).values, q=q, iterations=0, residual=float("nan"))


def compare_q(q_net, q_tab, tau: float) -> dict:
    """How close network Q-values (N, n_actions) are to tabular ones, for the same states."""
    adv_net = q_net - q_net.mean(dim=1, keepdim=True)
    adv_tab = q_tab - q_tab.mean(dim=1, keepdim=True)
    best = q_tab.max(dim=1, keepdim=True).values
    near_best = q_tab >= best - 1e-3
    greedy_ok = near_best.gather(1, q_net.argmax(dim=1, keepdim=True)).float().mean()
    log_p, log_q = torch.log_softmax(tau * q_tab, 1), torch.log_softmax(tau * q_net, 1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)
    tv = 0.5 * (log_p.exp() - log_q.exp()).abs().sum(dim=1)
    return {
        "q_mae": (q_net - q_tab).abs().mean().item(),
        "adv_mae": (adv_net - adv_tab).abs().mean().item(),
        "greedy": greedy_ok.item(),
        "kl": kl.mean().item(),
        "tv": tv.mean().item(),
        "tv_max": tv.max().item(),
    }
