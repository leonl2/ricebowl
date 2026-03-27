"""Programmatic explorer agent - no LLM required.

Uses grid analysis, change detection, and systematic exploration to play games.
Designed for competition use where LLM API calls are not available.
"""

import logging
import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()

# Movement actions and their direction vectors
MOVE_ACTIONS = {
    GameAction.ACTION1: (0, -1),   # Up
    GameAction.ACTION2: (0, 1),    # Down
    GameAction.ACTION3: (-1, 0),   # Left
    GameAction.ACTION4: (1, 0),    # Right
}

INTERACT_ACTION = GameAction.ACTION5


class GridAnalyzer:
    """Analyzes 64x64 game grids to extract structured information."""

    def __init__(self, grid: list[list[int]]) -> None:
        self.grid = np.array(grid, dtype=np.int8)
        self.h, self.w = self.grid.shape
        self._color_counts: Optional[dict[int, int]] = None
        self._bg_color: Optional[int] = None

    @property
    def bg_color(self) -> int:
        if self._bg_color is None:
            self._bg_color = int(np.argmax(np.bincount(self.grid.flatten())))
        return self._bg_color

    @property
    def color_counts(self) -> dict[int, int]:
        if self._color_counts is None:
            unique, counts = np.unique(self.grid, return_counts=True)
            self._color_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        return self._color_counts

    def find_objects(self) -> list[dict[str, Any]]:
        """Find connected non-background regions (objects) in the grid."""
        visited = np.zeros_like(self.grid, dtype=bool)
        objects = []

        for y in range(self.h):
            for x in range(self.w):
                color = int(self.grid[y, x])
                if color == self.bg_color or color == 0 or visited[y, x]:
                    continue

                # BFS to find connected component
                obj_cells: list[tuple[int, int]] = []
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if (
                        0 <= cx < self.w
                        and 0 <= cy < self.h
                        and not visited[cy, cx]
                        and int(self.grid[cy, cx]) == color
                    ):
                        visited[cy, cx] = True
                        obj_cells.append((cx, cy))
                        stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

                if len(obj_cells) >= 2:
                    xs = [c[0] for c in obj_cells]
                    ys = [c[1] for c in obj_cells]
                    objects.append({
                        "color": color,
                        "cells": obj_cells,
                        "size": len(obj_cells),
                        "bbox": (min(xs), min(ys), max(xs), max(ys)),
                        "center": (sum(xs) // len(xs), sum(ys) // len(ys)),
                    })

        return sorted(objects, key=lambda o: -o["size"])

    def diff(self, other: "GridAnalyzer") -> dict[str, Any]:
        """Compare this grid to another, identifying what changed."""
        if self.grid.shape != other.grid.shape:
            return {"error": "shape_mismatch"}

        changed_mask = self.grid != other.grid
        n_changed = int(np.sum(changed_mask))

        if n_changed == 0:
            return {"changed": False, "n_changed": 0}

        changed_coords = list(zip(*np.where(changed_mask)))
        # Identify what appeared and disappeared
        appeared: dict[int, list[tuple[int, int]]] = defaultdict(list)
        disappeared: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for y, x in changed_coords:
            old_val = int(self.grid[y, x])
            new_val = int(other.grid[y, x])
            if old_val != 0 and old_val != self.bg_color:
                disappeared[old_val].append((int(x), int(y)))
            if new_val != 0 and new_val != other.bg_color:
                appeared[new_val].append((int(x), int(y)))

        return {
            "changed": True,
            "n_changed": n_changed,
            "appeared": dict(appeared),
            "disappeared": dict(disappeared),
        }

    def get_status_rows(self) -> dict[str, list[int]]:
        """Extract bottom status rows (energy, score, info)."""
        result = {}
        if self.h >= 64:
            result["row_61"] = self.grid[61].tolist()
            result["row_62"] = self.grid[62].tolist()
            result["row_63"] = self.grid[63].tolist()
        return result


class Explorer(Agent):
    """Programmatic agent that explores games through systematic grid analysis.

    Strategy:
    1. Track grid changes after each action to learn what each action does
    2. Identify player position by finding what moves when we move
    3. Build a map of walls and walkable areas
    4. Systematically explore unmapped areas
    5. Interact with objects when discovered
    6. On GAME_OVER, reset and apply learned knowledge
    """

    MAX_ACTIONS: int = 150

    # State tracking
    prev_grid: Optional[GridAnalyzer]
    player_pos: Optional[tuple[int, int]]
    wall_cells: set[tuple[int, int]]
    visited_cells: set[tuple[int, int]]
    action_history: list[GameAction]
    action_effects: dict[str, int]  # action -> times it caused a change
    last_action: Optional[GameAction]
    stuck_count: int
    explore_direction: int
    lives_used: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_knowledge()

    def _reset_knowledge(self) -> None:
        self.prev_grid = None
        self.player_pos = None
        self.wall_cells = set()
        self.visited_cells = set()
        self.action_history = []
        self.action_effects = {a.name: 0 for a in MOVE_ACTIONS}
        self.action_effects[INTERACT_ACTION.name] = 0
        self.last_action = None
        self.stuck_count = 0
        self.explore_direction = 0
        self.lives_used = 0

    @property
    def name(self) -> str:
        return f"{super().name}.explorer"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        state = latest_frame.state

        # Must reset if not started or game over
        if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            if state == GameState.GAME_OVER:
                self.lives_used += 1
                logger.info(f"GAME_OVER (life #{self.lives_used}). Resetting with knowledge preserved.")
                # Keep wall knowledge but clear position
                self.player_pos = None
                self.prev_grid = None
            return GameAction.RESET

        # Analyze current grid
        current_grids = latest_frame.frame
        if not current_grids:
            return self._random_move()

        current = GridAnalyzer(current_grids[0])

        # Detect changes from last action
        if self.prev_grid is not None and self.last_action is not None:
            diff = self.prev_grid.diff(current)
            self._process_diff(diff, self.last_action, current)
        else:
            # First frame after reset - analyze initial state
            self._analyze_initial(current)

        self.prev_grid = current

        # Choose next action based on current knowledge
        action = self._decide_action(current, latest_frame)
        self.last_action = action
        self.action_history.append(action)

        action.reasoning = {
            "player_pos": self.player_pos,
            "stuck_count": self.stuck_count,
            "visited": len(self.visited_cells),
            "walls": len(self.wall_cells),
            "lives_used": self.lives_used,
            "strategy": "explore",
        }
        return action

    def _analyze_initial(self, grid: GridAnalyzer) -> None:
        """Analyze the initial game state to find player and objects."""
        objects = grid.find_objects()
        logger.info(f"Initial analysis: {len(objects)} objects found")
        for obj in objects[:5]:
            logger.info(
                f"  Object: color={obj['color']}, size={obj['size']}, "
                f"center={obj['center']}, bbox={obj['bbox']}"
            )

    def _process_diff(
        self, diff: dict[str, Any], action: GameAction, current: GridAnalyzer
    ) -> None:
        """Learn from how the grid changed after our action."""
        if not diff.get("changed", False):
            self.stuck_count += 1
            # If we tried to move and nothing changed, there's a wall
            if action in MOVE_ACTIONS and self.player_pos:
                dx, dy = MOVE_ACTIONS[action]
                wall_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                self.wall_cells.add(wall_pos)
                logger.debug(f"Wall detected at {wall_pos}")
            return

        self.stuck_count = 0
        self.action_effects[action.name] = self.action_effects.get(action.name, 0) + 1

        # Try to identify player position from movement
        if action in MOVE_ACTIONS:
            appeared = diff.get("appeared", {})
            disappeared = diff.get("disappeared", {})

            if appeared and disappeared:
                # Player likely moved: find the color that both appeared and disappeared
                for color in set(appeared.keys()) & set(disappeared.keys()):
                    new_positions = appeared[color]
                    old_positions = disappeared[color]
                    if new_positions and old_positions:
                        # Average of new positions is likely player location
                        avg_x = sum(p[0] for p in new_positions) // len(new_positions)
                        avg_y = sum(p[1] for p in new_positions) // len(new_positions)
                        self.player_pos = (avg_x, avg_y)
                        self.visited_cells.add(self.player_pos)
                        logger.info(f"Player detected at {self.player_pos} (color {color})")
                        break

    def _decide_action(self, grid: GridAnalyzer, frame: FrameData) -> GameAction:
        """Decide the next action based on accumulated knowledge."""
        move_actions = list(MOVE_ACTIONS.keys())

        # If stuck for too long, try interact or random direction
        if self.stuck_count >= 4:
            self.stuck_count = 0
            # Try interact first
            if self.action_effects.get(INTERACT_ACTION.name, 0) > 0 or random.random() < 0.3:
                return INTERACT_ACTION
            # Try a completely different direction
            self.explore_direction = (self.explore_direction + 2) % 4
            return move_actions[self.explore_direction]

        if self.stuck_count >= 2:
            # Stuck on this direction, try a different one
            self.explore_direction = (self.explore_direction + 1) % 4
            return move_actions[self.explore_direction]

        # If we know player position, do intelligent exploration
        if self.player_pos:
            return self._explore_from_position(grid)

        # No player position yet - cycle through directions to find player
        action = move_actions[self.action_counter % 4]
        return action

    def _explore_from_position(self, grid: GridAnalyzer) -> GameAction:
        """Explore systematically from known player position."""
        px, py = self.player_pos  # type: ignore[misc]

        # Find objects we haven't visited
        objects = grid.find_objects()
        interesting_objects = [
            obj for obj in objects
            if obj["center"] not in self.visited_cells
            and obj["color"] not in (grid.bg_color, 0, 10)  # skip bg and walls
            and obj["size"] < 100  # skip very large things (probably terrain)
        ]

        if interesting_objects:
            # Move toward the nearest interesting object
            target = min(
                interesting_objects,
                key=lambda o: abs(o["center"][0] - px) + abs(o["center"][1] - py),
            )
            tx, ty = target["center"]
            logger.info(
                f"Moving toward object: color={target['color']}, "
                f"pos={target['center']}, size={target['size']}"
            )
            return self._move_toward(px, py, tx, ty)

        # No interesting objects found - explore systematically
        # Use a spiral-like pattern from current position
        directions = list(MOVE_ACTIONS.keys())
        # Prefer the current exploration direction
        preferred = directions[self.explore_direction % 4]

        # Check if we'd hit a known wall
        dx, dy = MOVE_ACTIONS[preferred]
        next_pos = (px + dx, py + dy)
        if next_pos in self.wall_cells:
            # Rotate direction
            self.explore_direction = (self.explore_direction + 1) % 4
            return directions[self.explore_direction % 4]

        return preferred

    def _move_toward(self, px: int, py: int, tx: int, ty: int) -> GameAction:
        """Choose a movement action that moves us toward target."""
        dx = tx - px
        dy = ty - py

        # Prefer larger gap direction, avoid known walls
        candidates: list[tuple[GameAction, int]] = []

        if dx > 0:
            candidates.append((GameAction.ACTION4, abs(dx)))  # Right
        elif dx < 0:
            candidates.append((GameAction.ACTION3, abs(dx)))  # Left

        if dy > 0:
            candidates.append((GameAction.ACTION2, abs(dy)))  # Down
        elif dy < 0:
            candidates.append((GameAction.ACTION1, abs(dy)))  # Up

        # Filter out directions with known walls
        valid = []
        for action, priority in candidates:
            adx, ady = MOVE_ACTIONS[action]
            next_pos = (px + adx, py + ady)
            if next_pos not in self.wall_cells:
                valid.append((action, priority))

        if valid:
            # Pick direction with larger gap
            valid.sort(key=lambda x: -x[1])
            return valid[0][0]

        # All directions blocked, try interact or random
        if random.random() < 0.3:
            return INTERACT_ACTION
        return random.choice(list(MOVE_ACTIONS.keys()))

    def _random_move(self) -> GameAction:
        return random.choice(list(MOVE_ACTIONS.keys()))
