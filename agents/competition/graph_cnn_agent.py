"""Competition agent: State Graph + CNN Action Predictor.

Combines the top two approaches from the ARC-AGI-3 preview competition:
- 3rd place: Systematic graph-based exploration with component priority
- 1st place: CNN that learns which actions cause state changes

Pipeline per turn:
1. Hash the current frame, add to state graph
2. Record (prev_state, action, did_change) in CNN experience buffer
3. Periodically train the CNN on accumulated experience
4. Choose next action:
   a. If current state has untested actions, prefer CNN-predicted productive ones
   b. If current state is fully explored, navigate to nearest frontier state
   c. If CNN is confident, use its predictions to bias action selection
   d. Fall back to component-priority exploration

Designed for: RTX 5090, 8 hours, no internet access.
"""

import logging
import random
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .action_predictor import ActionPredictor
from .frame_analyzer import (
    Component,
    analyze_frame,
    compute_priority,
    detect_player_movement,
)
from .state_graph import GAME_ACTIONS, StateGraph, hash_frame

logger = logging.getLogger()

# Map action index to GameAction
ACTION_INDEX_MAP = {
    0: GameAction.ACTION1,  # Up
    1: GameAction.ACTION2,  # Down
    2: GameAction.ACTION3,  # Left
    3: GameAction.ACTION4,  # Right
    4: GameAction.ACTION5,  # Interact
    5: GameAction.ACTION6,  # Click
}

# Reverse map
ACTION_TO_INDEX = {v: k for k, v in ACTION_INDEX_MAP.items()}


class GraphCNNAgent(Agent):
    """Competition agent combining graph search with CNN action prediction.

    This is our best agent for the ARC Prize 2026 competition.
    """

    MAX_ACTIONS: int = 200

    # Tuning parameters
    CNN_TRAIN_INTERVAL: int = 10  # Train CNN every N actions
    CNN_TRAIN_STEPS: int = 10  # Steps per training round
    CNN_CONFIDENCE_THRESHOLD: float = 0.6  # Use CNN predictions above this
    EXPLORATION_EPSILON: float = 0.1  # Random exploration rate
    CLICK_EXPLORE_PROB: float = 0.15  # Probability of trying ACTION6

    # State
    graph: StateGraph
    predictor: ActionPredictor
    prev_frame: Optional[np.ndarray]
    prev_hash: Optional[str]
    prev_action: Optional[GameAction]
    player_color: Optional[int]
    player_pos: Optional[tuple[int, int]]
    current_level: int
    replay_queue: list[str]  # Action names to replay (navigating to frontier)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.graph = StateGraph()
        self.predictor = ActionPredictor()
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action = None
        self.player_color = None
        self.player_pos = None
        self.current_level = 0
        self.replay_queue = []

    @property
    def name(self) -> str:
        return f"{super().name}.graphcnn"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        state = latest_frame.state

        # Handle non-playing states
        if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            if state == GameState.GAME_OVER:
                logger.info(
                    f"GAME_OVER at turn {self.action_counter}. "
                    f"Graph: {self.graph.stats()}, CNN: {self.predictor.stats}"
                )
                self.graph.reset_path()
            self.prev_frame = None
            self.prev_hash = None
            self.replay_queue = []
            return GameAction.RESET

        # Get current grid
        if not latest_frame.frame:
            return self._random_action()

        current_grid = np.array(latest_frame.frame[0], dtype=np.int8)

        # Add state to graph
        current_hash = self.graph.add_state(current_grid, latest_frame.levels_completed)
        self.graph.set_current(current_hash)

        # Record transition from previous state
        did_change = True
        if self.prev_hash is not None and self.prev_action is not None:
            did_change = current_hash != self.prev_hash
            self.graph.record_transition(
                self.prev_hash, self.prev_action.name, current_hash
            )

            # Record in CNN buffer
            action_idx = ACTION_TO_INDEX.get(self.prev_action, 4)
            click_xy = None
            if self.prev_action == GameAction.ACTION6:
                data = self.prev_action.action_data
                if hasattr(data, "x") and hasattr(data, "y"):
                    click_xy = (data.x, data.y)
            self.predictor.record(
                self.prev_frame, action_idx, did_change, click_xy
            )

        # Detect player movement
        if self.prev_frame is not None:
            movement = detect_player_movement(self.prev_frame, current_grid)
            if movement:
                _, new_center, color = movement
                self.player_pos = new_center
                self.player_color = color

        # Check for level change -> reset CNN
        if latest_frame.levels_completed > self.current_level:
            logger.info(
                f"Level up! {self.current_level} -> {latest_frame.levels_completed}. "
                f"Resetting CNN."
            )
            self.current_level = latest_frame.levels_completed
            self.predictor.reset_model()
            # Keep buffer - some knowledge may transfer

        # Periodically train CNN
        if (
            self.action_counter > 0
            and self.action_counter % self.CNN_TRAIN_INTERVAL == 0
            and len(self.predictor.buffer) >= 32
        ):
            avg_loss = self.predictor.train_epoch(
                n_steps=self.CNN_TRAIN_STEPS, batch_size=min(64, len(self.predictor.buffer))
            )
            logger.info(
                f"CNN trained: loss={avg_loss:.4f}, buffer={len(self.predictor.buffer)}"
            )

        # Choose action
        action = self._select_action(current_grid, current_hash, latest_frame)

        # Store for next turn
        self.prev_frame = current_grid
        self.prev_hash = current_hash
        self.prev_action = action

        action.reasoning = {
            "graph": self.graph.stats(),
            "cnn": self.predictor.stats,
            "player_pos": self.player_pos,
            "level": self.current_level,
        }
        return action

    def _select_action(
        self, grid: np.ndarray, state_hash: str, frame: FrameData
    ) -> GameAction:
        """Core action selection logic."""

        # 1. If we have a replay queue (navigating to frontier), follow it
        if self.replay_queue:
            action_name = self.replay_queue.pop(0)
            logger.debug(f"Replaying: {action_name} ({len(self.replay_queue)} remaining)")
            return GameAction.from_name(action_name)

        node = self.graph.nodes.get(state_hash)
        if node is None:
            return self._random_action()

        # 2. Epsilon-greedy random exploration
        if random.random() < self.EXPLORATION_EPSILON:
            return self._random_action()

        # 3. If current state has untested actions, try them (CNN-guided)
        untested = node.untested_actions
        if untested:
            return self._choose_untested(grid, untested)

        # 4. Current state fully explored -> navigate to frontier
        frontier = self.graph.get_frontier_states()
        if frontier:
            target = frontier[0]
            path = self.graph.shortest_path(state_hash, target.state_hash)
            if path and len(path) > 0:
                self.replay_queue = path[1:]  # Queue up remaining actions
                logger.info(
                    f"Navigating to frontier state {target.state_hash[:8]} "
                    f"({len(path)} steps, {len(target.untested_actions)} untested actions)"
                )
                return GameAction.from_name(path[0])

        # 5. No reachable frontier - use CNN predictions
        return self._cnn_guided_action(grid)

    def _choose_untested(self, grid: np.ndarray, untested: list[str]) -> GameAction:
        """Choose among untested actions, using CNN to prioritize."""
        if len(self.predictor.buffer) < 32:
            # CNN hasn't trained enough - use component priority
            return self._priority_guided(grid, untested)

        # Get CNN predictions
        action_probs, spatial_map = self.predictor.predict(grid)

        # Score each untested action
        scored: list[tuple[str, float]] = []
        for action_name in untested:
            action = GameAction.from_name(action_name)
            idx = ACTION_TO_INDEX.get(action, -1)
            if 0 <= idx < 5:
                scored.append((action_name, float(action_probs[idx])))

        if not scored:
            return GameAction.from_name(random.choice(untested))

        # Sort by predicted change probability (prefer actions likely to have effect)
        scored.sort(key=lambda x: -x[1])

        # If top prediction is confident, use it
        if scored[0][1] > self.CNN_CONFIDENCE_THRESHOLD:
            logger.debug(f"CNN-guided: {scored[0][0]} (p={scored[0][1]:.2f})")
            return GameAction.from_name(scored[0][0])

        # Otherwise use the top one anyway (still better than random)
        return GameAction.from_name(scored[0][0])

    def _priority_guided(self, grid: np.ndarray, untested: list[str]) -> GameAction:
        """Use frame analysis to guide action selection when CNN isn't trained."""
        analysis = analyze_frame(grid)

        # If we have interactive components, try to move toward them
        if analysis.interactive and self.player_pos:
            px, py = self.player_pos
            target = analysis.interactive[0]
            tx, ty = target.center

            # Which direction gets us closer?
            dx, dy = tx - px, ty - py
            preferred = []
            if dy < 0 and "ACTION1" in untested:
                preferred.append("ACTION1")  # Up
            if dy > 0 and "ACTION2" in untested:
                preferred.append("ACTION2")  # Down
            if dx < 0 and "ACTION3" in untested:
                preferred.append("ACTION3")  # Left
            if dx > 0 and "ACTION4" in untested:
                preferred.append("ACTION4")  # Right

            if preferred:
                return GameAction.from_name(preferred[0])

        # Default: try actions in a systematic order
        priority_order = ["ACTION1", "ACTION4", "ACTION2", "ACTION3", "ACTION5"]
        for action_name in priority_order:
            if action_name in untested:
                return GameAction.from_name(action_name)

        return GameAction.from_name(random.choice(untested))

    def _cnn_guided_action(self, grid: np.ndarray) -> GameAction:
        """Use CNN predictions when no untested actions remain."""
        action_probs, spatial_map = self.predictor.predict(grid)

        # Should we try a click?
        max_spatial = float(np.max(spatial_map))
        if max_spatial > 0.7 or random.random() < self.CLICK_EXPLORE_PROB:
            # Find best click target
            y, x = np.unravel_index(np.argmax(spatial_map), spatial_map.shape)
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            logger.debug(f"CNN click: ({x}, {y}), confidence={max_spatial:.2f}")
            return action

        # Pick the simple action with highest predicted change probability
        best_idx = int(np.argmax(action_probs))
        action = ACTION_INDEX_MAP[best_idx]
        logger.debug(f"CNN action: {action.name} (p={action_probs[best_idx]:.2f})")
        return action

    def _random_action(self) -> GameAction:
        """Random action with slight bias toward movement."""
        if random.random() < self.CLICK_EXPLORE_PROB:
            action = GameAction.ACTION6
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            return action
        return random.choice(GAME_ACTIONS)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                self.recorder.record({
                    "agent_type": "graph_cnn",
                    "graph_stats": self.graph.stats(),
                    "cnn_stats": self.predictor.stats,
                    "player_color": self.player_color,
                    "final_level": self.current_level,
                })
        super().cleanup(*args, **kwargs)
