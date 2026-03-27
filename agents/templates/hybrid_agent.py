"""Hybrid agent: programmatic grid analysis + local LLM reasoning.

Combines the best of both approaches:
- GridAnalyzer extracts structured features (player pos, objects, walls, diffs)
- Local LLM receives compact, pre-processed information instead of raw grids
- Results in much better token efficiency and reasoning quality

This is the competition-ready agent. Designed to run on RTX 5090 with a local model.
"""

import json
import logging
import os
import re
import textwrap
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ..agent import Agent
from .explorer_agent import GridAnalyzer, MOVE_ACTIONS

logger = logging.getLogger()


class HybridAgent(Agent):
    """Competition agent: programmatic grid analysis feeds a local LLM for decisions.

    The LLM receives structured observations instead of raw grids, dramatically
    reducing token usage and improving reasoning quality.
    """

    MAX_ACTIONS: int = 120
    MESSAGE_LIMIT: int = 24

    # Analysis state
    prev_grid: Optional[GridAnalyzer]
    player_pos: Optional[tuple[int, int]]
    player_color: Optional[int]
    wall_cells: set[tuple[int, int]]
    visited_cells: set[tuple[int, int]]
    objects_seen: list[dict[str, Any]]
    action_results: list[dict[str, Any]]  # recent action->result log
    lives_used: int
    levels_seen: int

    # LLM state
    messages: list[dict[str, Any]]
    token_counter: int
    _latest_tool_call_id: str = "call_0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_state()

        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.environ.get("LOCAL_LLM_API_KEY", "ollama")
        self.model = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:32b-instruct")
        self.client = OpenAIClient(base_url=base_url, api_key=api_key)

    def _reset_state(self) -> None:
        self.prev_grid = None
        self.player_pos = None
        self.player_color = None
        self.wall_cells = set()
        self.visited_cells = set()
        self.objects_seen = []
        self.action_results = []
        self.lives_used = 0
        self.levels_seen = 0
        self.messages = []
        self.token_counter = 0

    @property
    def name(self) -> str:
        sanitized = self.model.replace("/", "-").replace(":", "-")
        return f"{super().name}.hybrid.{sanitized}"

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
                logger.info(f"GAME_OVER #{self.lives_used}. Resetting.")
                self._inject_game_over_learning()
            self.prev_grid = None
            self.player_pos = None
            return GameAction.RESET

        # Initialize LLM conversation on first real turn
        if len(self.messages) == 0:
            self.messages.append({
                "role": "system",
                "content": self._build_system_prompt(),
            })

        # Analyze current grid
        current_grids = latest_frame.frame
        if not current_grids:
            return GameAction.ACTION5

        current = GridAnalyzer(current_grids[0])

        # Extract structured analysis
        analysis = self._analyze_frame(current, latest_frame)

        # Update prev_grid for next diff
        self.prev_grid = current

        # Send structured analysis to LLM and get action
        action = self._llm_decide(analysis, latest_frame)

        action.reasoning = {
            "model": self.model,
            "player_pos": self.player_pos,
            "walls_known": len(self.wall_cells),
            "cells_visited": len(self.visited_cells),
            "lives_used": self.lives_used,
            "tokens_used": self.token_counter,
        }
        return action

    def _analyze_frame(self, grid: GridAnalyzer, frame: FrameData) -> str:
        """Run programmatic analysis and produce a compact text summary for the LLM."""
        parts = []

        # Header
        parts.append(
            f"Turn {self.action_counter} | State: {frame.state.name} | "
            f"Level: {frame.levels_completed}/{frame.win_levels} | "
            f"Lives used: {self.lives_used}"
        )

        # Diff analysis
        if self.prev_grid is not None:
            diff = self.prev_grid.diff(grid)
            if not diff.get("changed", False):
                parts.append("RESULT: Grid UNCHANGED (likely hit a wall or no effect)")
                if self.action_results:
                    last = self.action_results[-1]
                    parts.append(f"  Last action was: {last.get('action', '?')}")
            else:
                n = diff["n_changed"]
                parts.append(f"RESULT: {n} cells changed")

                appeared = diff.get("appeared", {})
                disappeared = diff.get("disappeared", {})

                if appeared:
                    parts.append(f"  Appeared: {dict((c, len(cells)) for c, cells in appeared.items())}")
                if disappeared:
                    parts.append(f"  Disappeared: {dict((c, len(cells)) for c, cells in disappeared.items())}")

                # Detect player movement
                self._detect_player(diff, grid)

                if frame.levels_completed > self.levels_seen:
                    parts.append(f"  ** LEVEL UP! Now at level {frame.levels_completed} **")
                    self.levels_seen = frame.levels_completed
        else:
            parts.append("FIRST FRAME after reset")

        # Player info
        if self.player_pos:
            parts.append(f"Player position: ({self.player_pos[0]}, {self.player_pos[1]}), color={self.player_color}")
            self.visited_cells.add(self.player_pos)

        # Find and report objects
        objects = grid.find_objects()
        notable_objects = [
            o for o in objects
            if o["color"] not in (grid.bg_color, 0)
            and o["size"] >= 4
            and o["size"] < 500
        ]

        if notable_objects:
            parts.append(f"Objects visible ({len(notable_objects)}):")
            for obj in notable_objects[:12]:
                dist = ""
                if self.player_pos:
                    d = abs(obj["center"][0] - self.player_pos[0]) + abs(obj["center"][1] - self.player_pos[1])
                    dist = f", dist={d}"
                parts.append(
                    f"  color={obj['color']}, size={obj['size']}, "
                    f"center=({obj['center'][0]},{obj['center'][1]}), "
                    f"bbox={obj['bbox']}{dist}"
                )

        # Status rows
        status = grid.get_status_rows()
        if status:
            for key, row in status.items():
                non_zero = [(i, v) for i, v in enumerate(row) if v != 0]
                if non_zero:
                    parts.append(f"{key}: {non_zero[:15]}")

        # Known walls near player
        if self.player_pos and self.wall_cells:
            px, py = self.player_pos
            nearby_walls = [
                (wx, wy)
                for wx, wy in self.wall_cells
                if abs(wx - px) <= 3 and abs(wy - py) <= 3
            ]
            if nearby_walls:
                parts.append(f"Known walls nearby: {nearby_walls[:10]}")

        # Exploration stats
        parts.append(f"Cells explored: {len(self.visited_cells)} | Walls mapped: {len(self.wall_cells)}")

        return "\n".join(parts)

    def _detect_player(self, diff: dict[str, Any], grid: GridAnalyzer) -> None:
        """Try to detect player position from grid diff."""
        appeared = diff.get("appeared", {})
        disappeared = diff.get("disappeared", {})

        # Player color: something that both appeared and disappeared (it moved)
        common_colors = set(appeared.keys()) & set(disappeared.keys())

        for color in common_colors:
            new_cells = appeared[color]
            old_cells = disappeared[color]
            if new_cells and old_cells and len(new_cells) < 30:
                avg_x = sum(p[0] for p in new_cells) // len(new_cells)
                avg_y = sum(p[1] for p in new_cells) // len(new_cells)

                old_avg_x = sum(p[0] for p in old_cells) // len(old_cells)
                old_avg_y = sum(p[1] for p in old_cells) // len(old_cells)

                # Movement should be small (1-4 cells)
                move_dist = abs(avg_x - old_avg_x) + abs(avg_y - old_avg_y)
                if 1 <= move_dist <= 8:
                    self.player_pos = (avg_x, avg_y)
                    self.player_color = color
                    return

    def _inject_game_over_learning(self) -> None:
        """Add a learning message when we die so the LLM avoids the same mistakes."""
        if self.messages:
            self.messages.append({
                "role": "user",
                "content": (
                    f"GAME OVER (death #{self.lives_used}). "
                    f"The game will reset. Remember what happened and avoid repeating mistakes. "
                    f"Last few actions: {[r.get('action', '?') for r in self.action_results[-5:]]}"
                ),
            })

    def _llm_decide(self, analysis: str, frame: FrameData) -> GameAction:
        """Send analysis to LLM and get an action decision."""
        self.messages.append({"role": "user", "content": analysis + "\n\nChoose ONE action."})

        tools = self._build_tools()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools,
                tool_choice="required",
                max_tokens=384,
            )
            self._track_tokens(response.usage.total_tokens if response.usage else 0)

            message = response.choices[0].message

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                action_name = tool_call.function.name
                self._latest_tool_call_id = tool_call.id

                try:
                    action_input = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    action_input = {}

                # Record in history
                self.messages.append(message)
                self.action_results.append({"action": action_name, "turn": self.action_counter})

                # Track walls from failed moves
                action = GameAction.from_name(action_name)
                if action_input:
                    action.set_data(action_input)

                logger.info(f"Hybrid LLM chose: {action_name}")
                return action

            # No tool call - try to parse from text
            if message.content:
                return self._parse_text_action(message.content)

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")

        # Fallback to programmatic exploration
        return self._fallback_action(frame)

    def _parse_text_action(self, text: str) -> GameAction:
        match = re.search(r"(RESET|ACTION[1-6])", text)
        if match:
            return GameAction.from_name(match.group(0))
        return self._fallback_action(None)

    def _fallback_action(self, frame: Optional[FrameData]) -> GameAction:
        """When LLM fails, fall back to systematic exploration."""
        import random

        move_list = list(MOVE_ACTIONS.keys())

        if self.player_pos:
            # Move away from walls toward unexplored areas
            px, py = self.player_pos
            for action in move_list:
                dx, dy = MOVE_ACTIONS[action]
                next_pos = (px + dx, py + dy)
                if next_pos not in self.wall_cells and next_pos not in self.visited_cells:
                    return action

        return random.choice(move_list)

    def _push_message(self, message: Any) -> None:
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            system = self.messages[:1]
            rest = self.messages[1:]
            trimmed = rest[-(self.MESSAGE_LIMIT - 1):]
            while trimmed and (
                trimmed[0].get("role") == "tool"
                if isinstance(trimmed[0], dict)
                else getattr(trimmed[0], "role", None) == "tool"
            ):
                trimmed.pop(0)
            self.messages = system + trimmed

    def _track_tokens(self, tokens: int) -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record({"tokens": tokens, "total_tokens": self.token_counter})
        logger.info(f"Tokens: +{tokens} = {self.token_counter}")

    def _build_system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are an expert ARC-AGI-3 game agent. You receive pre-analyzed game state
            and must choose one action per turn.

            You will receive:
            - Grid diff analysis (what changed after your last action)
            - Player position (if detected)
            - Nearby objects with colors, sizes, and distances
            - Status rows (energy, score)
            - Known walls and exploration stats

            ACTIONS: RESET (restart), ACTION1 (up), ACTION2 (down), ACTION3 (left),
            ACTION4 (right), ACTION5 (interact), ACTION6 (click x,y)

            KEY STRATEGIES:
            - If grid didn't change after a move, you hit a wall. Try another direction.
            - Track energy (row 61) - if it's dropping, find energy items (often color 6)
            - Move toward unexplored objects to learn what they do
            - When you find a door/exit, figure out what you need (key, color match, etc.)
            - Remember patterns across deaths - don't repeat the same mistakes
            - Color 10 = walls, Color 8 = floor, Color 11 = doors, Color 9 = rotators
            - Be systematic: clear one area before moving to the next""")

    def _build_tools(self) -> list[dict[str, Any]]:
        empty = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        return [
            {"type": "function", "function": {"name": "RESET", "description": "Restart game", "parameters": empty, "strict": True}},
            {"type": "function", "function": {"name": "ACTION1", "description": "Move UP", "parameters": empty, "strict": True}},
            {"type": "function", "function": {"name": "ACTION2", "description": "Move DOWN", "parameters": empty, "strict": True}},
            {"type": "function", "function": {"name": "ACTION3", "description": "Move LEFT", "parameters": empty, "strict": True}},
            {"type": "function", "function": {"name": "ACTION4", "description": "Move RIGHT", "parameters": empty, "strict": True}},
            {"type": "function", "function": {"name": "ACTION5", "description": "Interact", "parameters": empty, "strict": True}},
            {
                "type": "function",
                "function": {
                    "name": "ACTION6",
                    "description": "Click at coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "string", "description": "X (0-63)"},
                            "y": {"type": "string", "description": "Y (0-63)"},
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                self.recorder.record({
                    "model": self.model,
                    "total_tokens": self.token_counter,
                    "lives_used": self.lives_used,
                    "cells_visited": len(self.visited_cells),
                    "walls_mapped": len(self.wall_cells),
                })
        super().cleanup(*args, **kwargs)
