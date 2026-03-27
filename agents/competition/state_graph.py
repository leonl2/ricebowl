"""State graph for systematic environment exploration.

Inspired by the 3rd-place ARC-AGI-3 preview solution (graph-based exploration).
Builds a directed graph of game states connected by actions, enabling:
- Loop detection (avoid revisiting states)
- Frontier tracking (find states with untested actions)
- Shortest-path replay (efficiently revisit interesting states)
- Progress detection (track when score/levels change)
"""

import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from arcengine import GameAction

logger = logging.getLogger()

# All non-RESET actions the agent can take
GAME_ACTIONS = [
    GameAction.ACTION1,  # Up
    GameAction.ACTION2,  # Down
    GameAction.ACTION3,  # Left
    GameAction.ACTION4,  # Right
    GameAction.ACTION5,  # Interact
]
# ACTION6 (click) handled separately due to coordinate space


@dataclass
class StateNode:
    """A unique game state in the exploration graph."""

    state_hash: str
    frame: np.ndarray  # The raw grid (64x64)
    levels_completed: int = 0
    visit_count: int = 0

    # Edges: action -> resulting state hash (None = untested)
    transitions: dict[str, Optional[str]] = field(default_factory=dict)

    # Actions that caused no state change (walls/no-ops)
    dead_actions: set[str] = field(default_factory=set)

    # Actions that led to progress (score increase)
    good_actions: set[str] = field(default_factory=set)

    @property
    def untested_actions(self) -> list[str]:
        """Actions we haven't tried from this state."""
        all_actions = [a.name for a in GAME_ACTIONS]
        tested = set(self.transitions.keys()) | self.dead_actions
        return [a for a in all_actions if a not in tested]

    @property
    def is_fully_explored(self) -> bool:
        return len(self.untested_actions) == 0


def hash_frame(frame: np.ndarray, mask_status: bool = True) -> str:
    """Hash a game frame into a compact state identifier.

    Args:
        frame: 64x64 grid of color values (0-15)
        mask_status: If True, mask bottom rows (61-63) that often contain
                     changing status info (step counter, energy) to reduce
                     state explosion.
    """
    grid = frame.copy()
    if mask_status and grid.shape[0] >= 64:
        # Zero out status rows to avoid counting them as different states
        grid[61:, :] = 0
    return hashlib.md5(grid.tobytes()).hexdigest()[:16]


class StateGraph:
    """Directed graph of game states connected by actions.

    Supports:
    - Adding states and transitions
    - Finding frontier states (with untested actions)
    - Computing shortest paths between states
    - Detecting progress (level/score changes)
    """

    def __init__(self) -> None:
        self.nodes: dict[str, StateNode] = {}
        self.current_hash: Optional[str] = None

        # Track the path from RESET to current state (for replay)
        self.path_from_reset: list[tuple[str, str]] = []  # [(state_hash, action_name), ...]

        # Track best scores seen
        self.best_levels: int = 0

        # Statistics
        self.total_transitions: int = 0
        self.loops_detected: int = 0

    @property
    def current_node(self) -> Optional[StateNode]:
        if self.current_hash and self.current_hash in self.nodes:
            return self.nodes[self.current_hash]
        return None

    def add_state(self, frame: np.ndarray, levels_completed: int = 0) -> str:
        """Add a new state or return hash of existing one."""
        h = hash_frame(frame)
        if h not in self.nodes:
            self.nodes[h] = StateNode(
                state_hash=h,
                frame=frame.copy(),
                levels_completed=levels_completed,
            )
        node = self.nodes[h]
        node.visit_count += 1
        node.levels_completed = max(node.levels_completed, levels_completed)

        if levels_completed > self.best_levels:
            self.best_levels = levels_completed
            logger.info(f"New best level: {levels_completed}")

        return h

    def set_current(self, state_hash: str) -> None:
        """Set the current state."""
        self.current_hash = state_hash

    def record_transition(
        self, from_hash: str, action_name: str, to_hash: str
    ) -> bool:
        """Record a state transition. Returns True if the transition led to a new state."""
        if from_hash not in self.nodes or to_hash not in self.nodes:
            return False

        from_node = self.nodes[from_hash]
        self.total_transitions += 1

        if from_hash == to_hash:
            # Action had no effect (wall, no-op)
            from_node.dead_actions.add(action_name)
            return False

        # Check if this is a known loop
        if action_name in from_node.transitions and from_node.transitions[action_name] == to_hash:
            self.loops_detected += 1

        from_node.transitions[action_name] = to_hash

        # Check if transition led to progress
        to_node = self.nodes[to_hash]
        if to_node.levels_completed > from_node.levels_completed:
            from_node.good_actions.add(action_name)
            logger.info(
                f"Progress! {action_name} from {from_hash[:8]} increased level "
                f"{from_node.levels_completed} -> {to_node.levels_completed}"
            )

        return to_hash not in {t for t in from_node.transitions.values() if t is not None}

    def get_frontier_states(self) -> list[StateNode]:
        """Get all states that still have untested actions, sorted by promise."""
        frontier = [n for n in self.nodes.values() if not n.is_fully_explored]

        # Sort: prefer states with higher levels, fewer visits, more untested actions
        frontier.sort(
            key=lambda n: (
                -n.levels_completed,
                -len(n.untested_actions),
                n.visit_count,
            )
        )
        return frontier

    def shortest_path(self, from_hash: str, to_hash: str) -> Optional[list[str]]:
        """BFS to find shortest action sequence between two states."""
        if from_hash == to_hash:
            return []
        if from_hash not in self.nodes or to_hash not in self.nodes:
            return None

        visited = {from_hash}
        queue: deque[tuple[str, list[str]]] = deque([(from_hash, [])])

        while queue:
            current, path = queue.popleft()
            node = self.nodes[current]

            for action_name, next_hash in node.transitions.items():
                if next_hash and next_hash not in visited:
                    new_path = path + [action_name]
                    if next_hash == to_hash:
                        return new_path
                    visited.add(next_hash)
                    queue.append((next_hash, new_path))

        return None  # No path found

    def reset_path(self) -> None:
        """Clear the current path (called after RESET action)."""
        self.path_from_reset = []

    def stats(self) -> dict[str, int]:
        return {
            "states": len(self.nodes),
            "transitions": self.total_transitions,
            "loops": self.loops_detected,
            "frontier": len(self.get_frontier_states()),
            "best_level": self.best_levels,
        }
