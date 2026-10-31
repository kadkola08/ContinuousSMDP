# Project context (carried over from a previous Claude Code session)

Paste or point Claude at this file on a new machine to restore context. Last updated 2026-10-31.

## Working preferences

- **Skeleton first.** For the neural Social MDP, build end to end first. Don't run accuracy-improvement experiments or change the architecture or loss unless asked. Report accuracy honestly.
- **No config files.** Don't add `pyproject.toml` or other packaging or tool config to `pysmdp/` unless asked. Run tests as `python -m pytest` from `pysmdp/`.
- **Skip tests once a feature is verified.** If a feature is implemented and shown to work by running it, don't write a test suite for it unless asked.
- Original hardware: GTX 1660 Ti (6 GB), 15 GB RAM. Under load the GPU's memory clock stayed far below its maximum (810 of 6001 MHz, power state P5, software power cap and thermal slowdown), so bandwidth was about 30 GB/s. This was not addressed.

## Background

- **`tejwani22a.pdf`:** Tejwani et al., *Social Interactions as Recursive MDPs* (CoRL 2021). Social MDPs nest the *rewards* of the other agent.
  - The reward is r_i + Σ χ_ij R̃_j^{l−1} − c (Eq. 2), where χ > 0 means helping and χ < 0 hindering.
  - Level 0 is a plain MDP. Level 1 reasons about the other agent's physical goal. Level 2 reasons about its physical and social goals.
  - The other agent is predicted with a softmax policy over its Q (Eq. 7, τ). Goals are inferred by Bayes' rule (Eqs. 5 and 6).
- **`AmortizedNestedInference.pdf`:** Jha et al., *Neural Amortized Inference for Nested Multi-Agent Reasoning* (AAAI 2024). It uses one recognition network per level as a proposal, with importance sampling to weight the proposals, inside an I-POMDP. A weak proposal (17% accurate) was still enough.
- **The repo's original C++/CUDA solver** (`smdp.cu` and others) was ported to Python from scratch in `pysmdp/`.

## `pysmdp/` layout

**Tabular solver (`smdp/`):**
- `grid.py`: the 5×5 grid world with 2 robots and 2 cans; 5⁸ ≈ 390k states.
- `mdp.py`: rewards and `Params`.
- `solver.py`: value iteration at levels 0, 1 and 2.
- `policy.py`: softmax policies.
- `inference.py`: `GoalEstimator`.
- `agents.py`: tabular agents, with caching and belief rounding to 0.05.
- `episode.py`: the episode loop (Algorithm 1).
- `render.py`, `valueplot.py`: plotting.

**Neural version:**
- `smdp/neural.py`: Q-networks, training (`fit`), and the helpers `q_table`, `as_solution`, `load_levels` and `compare_q`.
- `smdp/neural_agents.py`: network-planning agents and `NeuralModels`.

**Scripts:**
- `run.py <config> [--neural out/neural]`: a full episode (records, filmstrip, GIF, belief plot).
- `run_l0.py`, `run_l1.py`, `run_l2.py`, `plot_values.py`, `run_social_inference.py`: demos. All accept `--neural <dir>`.
- `run_neural.py`: trains or evaluates the networks; writes `out/neural/q{0,1,2}.pt` and `report.json`.
- `make_figures.py`: regenerates every figure into `figures/tabular/` and `figures/neural/` (`--only`, `--out`, `--neural-dir`).
- `configs/`: `blocking_help`, `blocking_hinder` and `crossing` (`.json`).

**Tests:** `tests/`, 66 or more, including `test_neural.py` and `test_neural_agents.py`.

## Environment

- Physical reward: r_p(s, g) = max(ρ(1 − d(can_p, g)/δ), 0), with Euclidean distance.
- Action cost: 0.05 for a move, 0 for staying.
- `Params` defaults: γ = 0.95, ρ = 1, δ = 5, τ = 5. The paper uses ρ = 1.25, costs 1 and 0.1, γ = 0.99, and geodesic distance.
- Landmarks: flower (0, 1), tree (0, 3). Agent 0 (yellow) wants the flower; agent 1 (red) wants the tree.
- χ set: {−2, −1, −0.5, 0, 0.5, 1, 2}.

Reward at each level:
- Level 0: r_p
- Level 1: r_p + χ Σ_g b(g) r_j(g)
- Level 2: r_p + χ Σ_(g,χ_j) B(g, χ_j) [r_j(g) + χ_j r̂_p], with r̂_p = Σ b_j(g′) r_p(g′)

## Tabular profile (why it's slow)

- A level-2 episode on `blocking_help`, 10 steps, takes 88 s and 162 solves.
- About 84% of the time goes to re-solving the other agent's 14 level-1 hypotheses (2 goals × 7 χ) whenever its belief about the observer changes.
- Those hypotheses depend only on (agent, goal, χ, rounded belief). A precomputed library of them was proposed but not built.

## Neural Social MDP (built in this session)

**Design:**
- One Q-network per level, with the hypothesis as an input:
  - Q0(s, a; g)
  - Q1(s, a; g, χ, b)
  - Q2(s, a; g, χ, B, b_me)
- The network sees the state from the acting agent's point of view. Agent-swap symmetry was verified, so one network serves both agents.
- The rewards, dynamics and Bayesian inference stay exact.

**Inputs:**
- 37 state features: positions, pairwise offsets, the goal, and each can's offset and distance to the goal and to the landmarks.
- Level 1 adds own χ (provided by the config) and b (inferred; 2 numbers).
- Level 2 adds own χ, B (inferred joint belief over the other agent's (goal, χ); 14 numbers) and b_me (inferred by simulating the other agent's inference on the agent's own moves; 2 numbers).
- A level-1 agent infers only the other agent's goal. Only level 2 infers the other agent's χ.

**Training** (fitted value iteration, `fit()`):
- Each step:
  1. Sample 1024 valid states.
  2. Sample the hypotheses: χ uniform in [−2, 2], beliefs a mix of one-hot, sharp, flat and sparse.
  3. target = R(s) − c(a) + γ Σ_b ψ_j(s′, b) max_a′ Q_target(s″, a′). The other agent's policy ψ_j is uniform at level 0, otherwise the *frozen* network one level down, mixed over the belief.
  4. MSE loss on targets scaled by 1/(1−γ), Adam (1e-3, cosine to 2e-5), Polyak target update at rate 0.01.
- 20k steps per level. MLP 4×256, SiLU, dueling head; about 210k parameters per level.
- Levels are trained bottom-up. The lower network uses `requires_grad_(False)` and is not in the optimiser, so its errors carry upward.
- Training times: 10, 13 and 29 minutes. Level 2 calls the level-1 network for each of the 14 hypotheses.

**One network per level was trained once.** All figures, cases and both agents used `out/neural/q{0,1,2}.pt`; nothing was trained per case.

**Accuracy against value iteration** (averages over the evaluation cases, every valid state):

| Level | Cases | Q error | Action-difference error | Best action matches table | KL | TV |
|---|---|---|---|---|---|---|
| 0 | 2 | 0.31 | 0.07 | 76% | 0.08 | 0.11 |
| 1 | 5 | 2.19 | 0.26 | 61% | 0.52 | 0.30 |
| 2 | 3 | 1.81 | 0.30 | 46% | 0.65 | 0.32 |

TV is ½ Σ_a |π_tab − π_net|, averaged over states.

Evaluation cases:
- Level 0: yellow with goal flower, red with goal tree.
- Level 1: red with χ ∈ {+1, 0, −1} and certain that yellow wants the flower; red with χ = +2 and a 50/50 belief; yellow with χ = −0.5 and a 30/70 belief.
- Level 2: yellow with χ = 0, red as a certain (tree, +1) or a certain (tree, −1), each with b_me = flower; and a mixed belief 0.5 (tree, +1) / 0.3 (tree, −1) / 0.2 (flower, 0) with b_me = 70/30.

**Inference check** (blocking scenario, E[χ] at the end):

| Red's true χ | Tabular | Network |
|---|---|---|
| +1 | +1.43 | +1.95 |
| 0 | −0.50 | −0.06 |
| −1 | −1.71 | −0.68 |

The network gets the sign right but the size wrong.

**Full episode on `blocking_help`:**
- 1 s and 0 solves with networks, against 88 s and 162 solves with value iteration.
- Red helps in both runs. Yellow's E[χ] is similar: about +1.75 against +1.8, true +1. The overshoot is in the tabular run too.
- With networks, physical-goal inference is weaker (yellow ends at P(red's true goal) = 0.2, against 1.0), and yellow keeps pushing into the left wall after reaching its goal.

**Diagnosis, deferred on purpose:**
- The networks are imprecise about the small differences in Q between actions in a state, which softmax likelihoods are sensitive to.
- Fitting the network directly to the tabular Q, with a larger network or with a separately weighted action-difference loss, stalled at an action-difference error of about 0.07.
- Learning V and getting Q by one-step lookahead didn't help either.
- So the input representation is the likely limit. Fourier features were the next idea; the user stopped that experiment.

## Figures

- `make_figures.py` makes 96 files: 52 in `figures/tabular/` and 44 in `figures/neural/` with a `_neural` suffix. It takes about 11 minutes.
- Suggested for meetings:
  - `neural/values_agent0_neural.png`
  - `neural/social_inference_blocking_neural.png`
  - `neural/social_inference_blocking_chi-1_neural.gif`
  - `tabular/blocking_help.gif` and `neural/blocking_help_neural.gif`
  - `tabular/blocking_help_beliefs.png` and `neural/blocking_help_neural_beliefs.png`
  - optionally `neural/l2_blocking_redchi1_neural.gif`
- Known cosmetic issue: the legend overlaps the +1 curves in the inference summary figure.

## Ideas discussed, not built

- A network that outputs the other agent's mixed policy ψ(s; B, b_me) directly, replacing the 14 level-1 calls in level-2 training.
- A recognition network q(g, χ | trajectory), as in Jha et al.; useful for continuous χ or many goals.
- A set-encoder belief representation, and SAC for continuous actions.
- An exact policy library for the tabular solver: 2 × 7 × 21 level-1 policies per agent, about 1.2 GB in fp16.

## Next steps

1. Build a continuous 2D environment. Only the state sampler and the batched step function need replacing.
2. Later: improve accuracy, starting with the input representation.
3. Later: add the recognition-network proposals from Jha et al.

## Setting up on a new machine

- `pysmdp/` is **untracked** (nothing committed yet), so copy it, or commit and push it first.
- `pysmdp/out/` is gitignored, and it holds the trained networks in `out/neural/`. Copy that folder, or retrain with `python run_neural.py` (about 50 minutes on a 1660 Ti).
- Python packages: torch 2.14.1 (CUDA 13.0 build), numpy, matplotlib, pillow, pytest. On the original machine they were in `.venv/` at the repo root.
- Run everything from `pysmdp/`: `python run.py configs/blocking_help.json --neural out/neural`, `python make_figures.py`, `python -m pytest tests -q`.
