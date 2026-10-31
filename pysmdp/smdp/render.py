"""Drawing grid states as text or with matplotlib.

Agent 0 is drawn as the yellow robot and agent 1 as the red robot, each with its own can,
as in the paper's figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from smdp.grid import ACTIONS, N_ENTITIES, Grid, State, position

ACTION_NAMES = ("stay", "right", "left", "down", "up")
TEXT_SYMBOLS = ("Y", "R", "y", "r")  # yellow robot, red robot, yellow can, red can
LANDMARK_SYMBOLS = ("F", "T")  # flower, tree
IMAGE_DIR = Path(__file__).resolve().parents[2] / "images"
IMAGE_FILES = ("yellow_robot.png", "red_robot.png", "yellow_bucket.png", "red_bucket.png")
LANDMARK_FILES = ("flower.png", "tree.png")


def render_text(grid: Grid, s: State, landmarks=()) -> str:
    cells = [["." for _ in range(grid.n_side)] for _ in range(grid.n_side)]
    for (row, col), symbol in zip(landmarks, LANDMARK_SYMBOLS):
        cells[row][col] = symbol
    for entity in range(N_ENTITIES):
        row, col = position(s, entity)
        cells[row][col] = TEXT_SYMBOLS[entity]
    return "\n".join(" ".join(row) for row in cells)


def load_images():
    return [plt.imread(IMAGE_DIR / name) for name in IMAGE_FILES + LANDMARK_FILES]


def draw_image(ax, image, row, col, pad, alpha=1.0):
    ax.imshow(image, extent=(col + pad, col + 1 - pad, row + 1 - pad, row + pad), alpha=alpha)


def draw_state(ax, grid: Grid, s: State, images=None, title: str = "", landmarks=()):
    """Draws a state; ``landmarks`` are the goal cells, drawn as the flower and the tree."""
    images = images if images is not None else load_images()
    ax.clear()
    for i, (row, col) in enumerate(landmarks):
        draw_image(ax, images[N_ENTITIES + i], row, col, pad=0.05, alpha=0.6)
    for entity in range(N_ENTITIES):
        row, col = position(s, entity)
        draw_image(ax, images[entity], row, col, pad=0.1)
    ax.set_xlim(0, grid.n_side)
    ax.set_ylim(grid.n_side, 0)
    ax.set_xticks(range(grid.n_side + 1))
    ax.set_yticks(range(grid.n_side + 1))
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(True, color="0.85")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def step_title(t: int, actions: tuple[int, int] | None, blocked: tuple[bool, bool]) -> str:
    if actions is None:
        return "t = 0 (start)"
    names = [
        ACTION_NAMES[a] + (" (blocked)" if b else "") for a, b in zip(actions, blocked)
    ]
    return f"t = {t}\nY: {names[0]}\nR: {names[1]}"


def save_filmstrip(grid: Grid, states, titles, path, columns: int = 5, landmarks=()):
    rows = (len(states) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 3.0 * rows))
    images = load_images()
    for ax in axes.flat[len(states) :]:
        ax.axis("off")
    for ax, s, title in zip(axes.flat, states, titles):
        draw_state(ax, grid, s, images, title, landmarks)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_gif(grid: Grid, states, titles, path, fps: float = 1.5, landmarks=()):
    fig, ax = plt.subplots(figsize=(4, 4.6))
    images = load_images()

    def update(i):
        draw_state(ax, grid, states[i], images, titles[i], landmarks)

    animation = FuncAnimation(fig, update, frames=len(states))
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def is_blocked(before: State, after: State, action: int) -> bool:
    return ACTIONS[action] != (0, 0) and before == after


def save_comparison(grid: Grid, rows, path, landmarks=()):
    """Filmstrips stacked as rows, one per (label, states, titles), with equal frame counts."""
    n_frames = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), n_frames, figsize=(1.9 * n_frames, 2.5 * len(rows)), squeeze=False)
    images = load_images()
    for r, (label, states, titles) in enumerate(rows):
        for c, (s, title) in enumerate(zip(states, titles)):
            draw_state(axes[r, c], grid, s, images, title, landmarks)
            axes[r, c].title.set_fontsize(7)
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
