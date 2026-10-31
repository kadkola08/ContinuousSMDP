"""Value-function figures for one agent: a 2-D slice of the 8-D state space.

Everything except agent p is held fixed, and agent p is placed on each cell in turn. Each cell
then shows the value of the resulting state (``draw_values``, with an arrow for the best
action) or the Q-value of each action (``draw_q``: four triangles for the moves and a small
center square for staying). Cells holding another robot or a can are gray.
"""

import math

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

from smdp.grid import ACTIONS, DOWN, LEFT, N_ACTIONS, N_ENTITIES, RIGHT, STAY, UP, State, position
from smdp.render import LANDMARK_FILES
from smdp.solver import Solution

# Single-hue sequential ramp: light = low value, dark = high value.
VALUE_CMAP = LinearSegmentedColormap.from_list(
    "value_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
INK = "#1f1f1f"
OCCUPIED = "#e4e3df"
SURFACE = "#ffffff"
ARROW_GLYPHS = {RIGHT: "▸", LEFT: "◂", DOWN: "▾", UP: "▴", STAY: "•"}


def agent_slice(grid, solution: Solution, s: State, p: int):
    """Returns V and Q for agent p on every cell, others fixed as in s.

    V has shape (n, n) and Q has shape (n, n, n_actions); occupied cells are NaN.
    """
    n = grid.n_side
    V = np.full((n, n), np.nan)
    Q = np.full((n, n, N_ACTIONS), np.nan)
    for row in range(n):
        for col in range(n):
            moved = list(s)
            moved[2 * p], moved[2 * p + 1] = row, col
            moved = tuple(moved)
            if not grid.is_valid(moved):
                continue
            index = grid.encode(moved)
            V[row, col] = solution.values[index].item()
            Q[row, col] = solution.q[:, index].numpy()
    return V, Q


def value_norm(*arrays) -> Normalize:
    """A shared color scale for several slices, so colors are comparable between them."""
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    return Normalize(vmin=finite.min(), vmax=finite.max())


def _text_color(norm, value):
    return SURFACE if norm(value) > 0.55 else INK


def _draw_frame(ax, n, s, p, images, landmarks, title):
    """Occupied cells, landmark markers, the agent's actual cell, and axes styling."""
    for entity in range(N_ENTITIES):
        if entity == p:
            continue
        row, col = position(s, entity)
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=OCCUPIED, edgecolor=SURFACE, lw=2))
        ax.imshow(images[entity], extent=(col + 0.15, col + 0.85, row + 0.85, row + 0.15), zorder=3)
    for i, (row, col) in enumerate(landmarks[: len(LANDMARK_FILES)]):
        # A small landmark icon in the cell's top-left corner, so the numbers stay readable.
        ax.imshow(
            images[N_ENTITIES + i], extent=(col + 0.03, col + 0.3, row + 0.3, row + 0.03), zorder=4
        )
    row, col = position(s, p)
    ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor=INK, lw=2.5, zorder=5))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def draw_values(ax, s, p, V, Q, norm, images, landmarks=(), title=""):
    """V per cell, with a glyph for the best action next to the number."""
    ax.clear()
    n = V.shape[0]
    for row in range(n):
        for col in range(n):
            value = V[row, col]
            if math.isnan(value):
                continue
            ax.add_patch(
                Rectangle((col, row), 1, 1, facecolor=VALUE_CMAP(norm(value)), edgecolor=SURFACE, lw=2)
            )
            color = _text_color(norm, value)
            ax.text(col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color)
            best = int(np.argmax(Q[row, col]))
            dr, dc = ACTIONS[best]
            ax.text(
                col + 0.5 + 0.36 * dc,
                row + 0.5 + 0.36 * dr + (0.24 if best == STAY else 0),
                ARROW_GLYPHS[best],
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    _draw_frame(ax, n, s, p, images, landmarks, title)


def _q_shapes(row, col):
    """Polygon corners for each action's region of a cell."""
    tl, tr, bl, br = (col, row), (col + 1, row), (col, row + 1), (col + 1, row + 1)
    c = (col + 0.5, row + 0.5)
    h = 0.15  # half-size of the center "stay" square
    return {
        UP: [tl, tr, c],
        DOWN: [bl, br, c],
        LEFT: [tl, bl, c],
        RIGHT: [tr, br, c],
        STAY: [(c[0] - h, c[1] - h), (c[0] + h, c[1] - h), (c[0] + h, c[1] + h), (c[0] - h, c[1] + h)],
    }


def draw_q(ax, s, p, Q, norm, images, landmarks=(), title=""):
    """Q(s, a) per cell: one triangle per move and a center square for staying.

    The best action's value is in bold. Values below the color scale take its lightest color.
    """
    ax.clear()
    n = Q.shape[0]
    for row in range(n):
        for col in range(n):
            if np.isnan(Q[row, col]).any():
                continue
            best = int(np.argmax(Q[row, col]))
            shapes = _q_shapes(row, col)
            for a in (UP, DOWN, LEFT, RIGHT, STAY):
                value = Q[row, col, a]
                ax.add_patch(
                    Polygon(shapes[a], facecolor=VALUE_CMAP(norm(value)), edgecolor=SURFACE, lw=1.5)
                )
                dr, dc = ACTIONS[a]
                ax.text(
                    col + 0.5 + 0.36 * dc,
                    row + 0.5 + 0.36 * dr,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if a == best else "normal",
                    color=_text_color(norm, value),
                    zorder=2,
                )
    _draw_frame(ax, n, s, p, images, landmarks, title)


def policy_slice(grid, policy, s: State):
    """Action probabilities of policy.agent on every cell, others fixed as in s.

    Shape (n, n, n_actions); occupied cells are NaN.
    """
    n, p = grid.n_side, policy.agent
    P = np.full((n, n, N_ACTIONS), np.nan)
    for row in range(n):
        for col in range(n):
            moved = list(s)
            moved[2 * p], moved[2 * p + 1] = row, col
            moved = tuple(moved)
            if grid.is_valid(moved):
                P[row, col] = policy.probs_at(grid, moved)
    return P


def draw_policy(ax, s, p, P, images, landmarks=(), title=""):
    """One arrow per move with length proportional to its probability, and a dot for staying
    whose area is proportional to its probability."""
    ax.clear()
    n = P.shape[0]
    for row in range(n):
        for col in range(n):
            probs = P[row, col]
            if np.isnan(probs).any():
                continue
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#f4f8fe", edgecolor=SURFACE, lw=2))
            x, y = col + 0.5, row + 0.5
            for a in (UP, DOWN, LEFT, RIGHT):
                length = 0.45 * probs[a]
                if length < 0.02:
                    continue
                dr, dc = ACTIONS[a]
                ax.arrow(
                    x, y, dc * length, dr * length, width=0.025, head_width=0.09,
                    head_length=min(0.08, length), length_includes_head=True,
                    color="#256abf", zorder=3,
                )
            ax.scatter([x], [y], s=250 * probs[STAY], color=INK, zorder=2)
    _draw_frame(ax, n, s, p, images, landmarks, title)
