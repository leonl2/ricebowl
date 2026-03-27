"""Frame analysis: segment grids into components and prioritize interactive elements.

Inspired by the 3rd-place solution's approach of segmenting frames into
single-color connected components and using size/color/position heuristics
to identify interactive elements worth exploring.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger()


@dataclass
class Component:
    """A connected region of same-colored cells."""

    color: int
    cells: list[tuple[int, int]]  # (x, y) coordinates
    size: int
    bbox: tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    center: tuple[int, int]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def is_square(self) -> bool:
        return abs(self.width - self.height) <= 1

    @property
    def area_ratio(self) -> float:
        """Ratio of actual cells to bounding box area. 1.0 = solid rectangle."""
        box_area = self.width * self.height
        return self.size / box_area if box_area > 0 else 0


@dataclass
class FrameAnalysis:
    """Complete analysis of a single game frame."""

    grid: np.ndarray
    bg_color: int
    components: list[Component]

    # Detected elements
    player: Optional[Component] = None
    walls: list[Component] = None  # type: ignore[assignment]
    interactive: list[Component] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.walls is None:
            self.walls = []
        if self.interactive is None:
            self.interactive = []


def segment_frame(grid: np.ndarray) -> list[Component]:
    """Segment a 64x64 grid into connected components of same color.

    Uses flood-fill (BFS) for connected component labeling.
    Ignores color 0 (typically transparent/empty).
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if visited[y, x] or color == 0:
                continue

            # BFS flood fill
            cells: list[tuple[int, int]] = []
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if (
                    0 <= cx < w
                    and 0 <= cy < h
                    and not visited[cy, cx]
                    and int(grid[cy, cx]) == color
                ):
                    visited[cy, cx] = True
                    cells.append((cx, cy))
                    stack.extend([
                        (cx + 1, cy), (cx - 1, cy),
                        (cx, cy + 1), (cx, cy - 1),
                    ])

            if cells:
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                components.append(Component(
                    color=color,
                    cells=cells,
                    size=len(cells),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    center=(sum(xs) // len(xs), sum(ys) // len(ys)),
                ))

    return components


def find_background_color(grid: np.ndarray) -> int:
    """Identify the background color (most common non-zero color)."""
    flat = grid.flatten()
    # Exclude 0 (transparent)
    non_zero = flat[flat > 0]
    if len(non_zero) == 0:
        return 0
    counts = np.bincount(non_zero)
    return int(np.argmax(counts))


def compute_priority(component: Component, bg_color: int, grid_h: int = 64) -> float:
    """Score a component's likelihood of being interactive/important.

    Higher priority = more likely to be an interactive element worth exploring.

    Heuristics based on 3rd-place solution insights:
    - Small-to-medium size (4-100 cells) = likely interactive
    - Square/rectangular shape = likely a button/object
    - Not the background color
    - Not at grid edges (status bars)
    - Distinctive color (not walls = color 10)
    """
    score = 0.0

    # Size: prefer small-to-medium objects
    if 4 <= component.size <= 16:
        score += 3.0  # Small objects are often interactive
    elif 16 < component.size <= 64:
        score += 2.0
    elif 64 < component.size <= 200:
        score += 1.0
    elif component.size > 500:
        score -= 2.0  # Too large, probably terrain

    # Shape: square objects are often buttons/items
    if component.is_square and component.width <= 6:
        score += 2.0

    # Solid fill ratio: buttons tend to be solid
    if component.area_ratio > 0.8:
        score += 1.0

    # Color significance
    if component.color == bg_color:
        score -= 5.0  # Background
    if component.color == 10:
        score -= 3.0  # Walls
    if component.color in (6, 9, 11):
        score += 2.0  # Common interactive colors (energy, rotator, door)

    # Position: penalize bottom status rows
    if component.bbox[1] >= grid_h - 4:
        score -= 4.0

    # Position: penalize very edge positions
    if component.bbox[0] == 0 or component.bbox[2] >= 63:
        score -= 1.0

    return score


def analyze_frame(grid: np.ndarray) -> FrameAnalysis:
    """Full analysis of a game frame."""
    bg_color = find_background_color(grid)
    components = segment_frame(grid)

    # Classify components
    walls = []
    interactive = []
    player_candidates = []

    for comp in components:
        priority = compute_priority(comp, bg_color, grid.shape[0])

        if comp.color == 10 and comp.size > 20:
            walls.append(comp)
        elif priority > 2.0:
            interactive.append(comp)

        # Player detection: small, non-wall, non-background
        if (
            4 <= comp.size <= 20
            and comp.color not in (0, bg_color, 10)
            and comp.bbox[1] < 60  # Not in status bar
        ):
            player_candidates.append(comp)

    # Sort interactive by priority
    interactive.sort(key=lambda c: -compute_priority(c, bg_color))

    # Player is usually the moving object - we'll confirm via diff later
    # For now, pick the most "player-like" candidate
    player = None
    if player_candidates:
        # Prefer smaller, non-wall objects in the playable area
        player_candidates.sort(key=lambda c: c.size)
        player = player_candidates[0]

    return FrameAnalysis(
        grid=grid,
        bg_color=bg_color,
        components=components,
        player=player,
        walls=walls,
        interactive=interactive,
    )


def diff_frames(
    old_grid: np.ndarray, new_grid: np.ndarray
) -> dict[str, list[tuple[int, int]]]:
    """Find cells that changed between two frames.

    Returns dict with 'changed', 'appeared' (new non-zero), 'disappeared' (was non-zero).
    """
    changed_mask = old_grid != new_grid
    changed_coords = list(zip(*np.where(changed_mask)))

    appeared = []
    disappeared = []

    for y, x in changed_coords:
        old_val = int(old_grid[y, x])
        new_val = int(new_grid[y, x])
        if new_val != 0 and old_val == 0:
            appeared.append((int(x), int(y)))
        elif old_val != 0 and new_val == 0:
            disappeared.append((int(x), int(y)))

    return {
        "changed": [(int(x), int(y)) for y, x in changed_coords],
        "appeared": appeared,
        "disappeared": disappeared,
        "n_changed": int(np.sum(changed_mask)),
    }


def detect_player_movement(
    old_grid: np.ndarray, new_grid: np.ndarray
) -> Optional[tuple[tuple[int, int], tuple[int, int], int]]:
    """Detect if a small object moved between frames.

    Returns (old_center, new_center, color) if movement detected, else None.
    """
    diff_mask = old_grid != new_grid
    if not np.any(diff_mask):
        return None

    # Find colors that both appeared and disappeared (moved objects)
    changed_ys, changed_xs = np.where(diff_mask)

    # Group changes by what the old value was and new value is
    old_vals = old_grid[changed_ys, changed_xs]
    new_vals = new_grid[changed_ys, changed_xs]

    # Find colors present in both old and new changed cells
    old_colors = set(int(v) for v in old_vals if v != 0)
    new_colors = set(int(v) for v in new_vals if v != 0)
    moved_colors = old_colors & new_colors

    for color in moved_colors:
        # Old positions of this color
        old_mask = (old_grid == color) & diff_mask
        new_mask = (new_grid == color) & diff_mask

        old_ys, old_xs = np.where(old_mask)
        new_ys, new_xs = np.where(new_mask)

        if len(old_xs) == 0 or len(new_xs) == 0:
            continue

        # Small clusters only (likely player, not terrain)
        if len(old_xs) > 25 or len(new_xs) > 25:
            continue

        old_center = (int(np.mean(old_xs)), int(np.mean(old_ys)))
        new_center = (int(np.mean(new_xs)), int(np.mean(new_ys)))

        # Movement should be small
        dist = abs(new_center[0] - old_center[0]) + abs(new_center[1] - old_center[1])
        if 1 <= dist <= 8:
            return (old_center, new_center, color)

    return None
