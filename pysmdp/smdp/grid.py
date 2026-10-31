from dataclasses import dataclass

import torch

N_COMPONENTS = 8
N_ENTITIES = 4
AGENTS = (0, 1)
CANS = (2, 3)

STAY, RIGHT, LEFT, DOWN, UP = range(5)
# (row change, col change) for each action.
ACTIONS = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))
N_ACTIONS = len(ACTIONS)

State = tuple[int, ...]


def other_agent(p: int) -> int:
    return 1 - p


def can_of(p: int) -> int:
    return p + 2


def make_state(agent0, agent1, can0, can1) -> State:
    """Builds a state from four (row, col) pairs."""
    return (*agent0, *agent1, *can0, *can1)


def position(s: State, entity: int) -> tuple[int, int]:
    return s[2 * entity], s[2 * entity + 1]


@dataclass(frozen=True)
class Grid:
    n_side: int

    @property
    def n_states(self) -> int:
        return self.n_side**N_COMPONENTS

    # ----- Scalar versions -----

    def encode(self, s: State) -> int:
        index = 0
        for c in s:
            index = index * self.n_side + c
        return index

    def decode(self, index: int) -> State:
        s = [0] * N_COMPONENTS
        for i in reversed(range(N_COMPONENTS)):
            index, s[i] = divmod(index, self.n_side)
        return tuple(s)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n_side and 0 <= col < self.n_side

    @staticmethod
    def is_valid(s: State) -> bool:
        """A state is valid if no two entities share a cell."""
        return len({position(s, e) for e in range(N_ENTITIES)}) == N_ENTITIES

    def step(self, s: State, p: int, action: int) -> State:
        """Moves agent ``p`` by ``action``, pushing cans as needed."""
        dr, dc = ACTIONS[action]
        if (dr, dc) == (0, 0):
            return s
        blocker = position(s, other_agent(p))
        new = list(s)

        def advance(entity: int, row: int, col: int) -> tuple[int, int] | None:
            row, col = row + dr, col + dc
            if not self.in_bounds(row, col) or (row, col) == blocker:
                return None
            new[2 * entity], new[2 * entity + 1] = row, col
            return row, col

        target = advance(p, *position(s, p))
        if target is None:
            return s
        for can in CANS:
            if position(s, can) != target:
                continue
            can_target = advance(can, *target)
            if can_target is None:
                return s
            second = CANS[0] if can == CANS[1] else CANS[1]
            if position(s, second) == can_target and advance(second, *can_target) is None:
                return s
            break
        return tuple(new)

    # ----- Batched versions, on (N, 8) integer tensors -----

    def decode_range(self, start: int, stop: int, device=None) -> torch.Tensor:
        """Decodes the states with indices in [start, stop) into an (N, 8) tensor."""
        index = torch.arange(start, stop, dtype=torch.long, device=device)
        states = torch.empty((stop - start, N_COMPONENTS), dtype=torch.long, device=device)
        for i in reversed(range(N_COMPONENTS)):
            states[:, i] = index % self.n_side
            index = index // self.n_side
        return states

    def encode_batch(self, states: torch.Tensor) -> torch.Tensor:
        index = torch.zeros(states.shape[0], dtype=torch.long, device=states.device)
        for i in range(N_COMPONENTS):
            index = index * self.n_side + states[:, i]
        return index

    @staticmethod
    def is_valid_batch(states: torch.Tensor) -> torch.Tensor:
        pos = states.view(-1, N_ENTITIES, 2)
        valid = torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
        for i in range(N_ENTITIES):
            for j in range(i + 1, N_ENTITIES):
                valid &= (pos[:, i] != pos[:, j]).any(dim=1)
        return valid

    def step_batch(self, states: torch.Tensor, p: int, action: int) -> torch.Tensor:
        """Batched ``step``. Only meaningful for valid states."""
        d = torch.tensor(ACTIONS[action], dtype=states.dtype, device=states.device)
        if not d.any():
            return states.clone()
        pos = states.view(-1, N_ENTITIES, 2)
        agent = pos[:, p]
        blocker = pos[:, other_agent(p)]
        can0, can1 = pos[:, CANS[0]], pos[:, CANS[1]]

        def same(a, b):
            return (a == b).all(dim=1)

        def free(cell):
            inside = ((cell >= 0) & (cell < self.n_side)).all(dim=1)
            return inside & ~same(cell, blocker)

        # Cells reached by the agent, the first pushed can, and the second pushed can.
        t1, t2, t3 = agent + d, agent + 2 * d, agent + 3 * d
        hit0, hit1 = same(can0, t1), same(can1, t1)
        chain0 = hit0 & same(can1, t2)  # can0 pushed into can1
        chain1 = hit1 & same(can0, t2)  # can1 pushed into can0
        hit, chain = hit0 | hit1, chain0 | chain1

        ok = free(t1) & (~hit | free(t2)) & (~chain | free(t3))

        new = pos.clone()
        new[:, p] = torch.where(ok[:, None], t1, agent)
        move0 = torch.where((ok & chain1)[:, None], t3, can0)
        new[:, CANS[0]] = torch.where((ok & hit0)[:, None], t2, move0)
        move1 = torch.where((ok & chain0)[:, None], t3, can1)
        new[:, CANS[1]] = torch.where((ok & hit1)[:, None], t2, move1)
        return new.view(-1, N_COMPONENTS)
