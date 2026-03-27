import json
import logging
import os
import textwrap
from typing import Any, Optional

import anthropic
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


class ClaudeAgent(Agent):
    """An ARC-AGI-3 agent powered by Claude (Anthropic).

    Uses Claude's tool_use capability to select game actions.
    Maintains conversation history with observations and reasoning.
    """

    MAX_ACTIONS: int = 80
    MODEL: str = "claude-sonnet-4-20250514"
    MESSAGE_LIMIT: int = 20  # keep more context than OpenAI agents
    DO_OBSERVATION: bool = True

    messages: list[dict[str, Any]]
    token_counter: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    @property
    def name(self) -> str:
        sanitized = self.MODEL.replace("/", "-").replace(":", "-")
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        return f"{super().name}.{sanitized}.{obs}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        tools = self._build_tools()

        # First turn: trigger RESET to start the game
        if len(self.messages) == 0:
            self._push_message(
                {
                    "role": "user",
                    "content": self._build_system_context(),
                }
            )
            self._push_message(
                {
                    "role": "assistant",
                    "content": "I'll start by resetting the game to begin playing. Let me observe the initial state.",
                }
            )
            return GameAction.RESET

        # Build observation from the latest frame
        observation = self._build_observation(latest_frame)
        self._push_message({"role": "user", "content": observation})

        # If doing observation, first ask Claude to reason about the state
        if self.DO_OBSERVATION:
            logger.info("Asking Claude to observe and reason...")
            try:
                obs_response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=1024,
                    messages=self.messages,
                )
                self._track_tokens(obs_response.usage.input_tokens + obs_response.usage.output_tokens)
                obs_text = obs_response.content[0].text if obs_response.content else ""
                logger.info(f"Claude observation: {obs_text[:200]}")
                self._push_message({"role": "assistant", "content": obs_text})
            except Exception as e:
                logger.warning(f"Observation call failed: {e}")

        # Now ask Claude to pick an action using tools
        self._push_message(
            {
                "role": "user",
                "content": "Based on your analysis, choose exactly ONE action to take next. Call one of the action tools.",
            }
        )

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=1024,
                tools=tools,
                tool_choice={"type": "any"},
                messages=self.messages,
            )
            self._track_tokens(response.usage.input_tokens + response.usage.output_tokens)
        except Exception as e:
            logger.error(f"Claude action call failed: {e}")
            # Fallback to ACTION5
            action = GameAction.ACTION5
            action.reasoning = f"Fallback due to API error: {e}"
            return action

        # Parse the tool call from the response
        action_name = "ACTION5"
        action_input: dict[str, Any] = {}
        reasoning_text = ""

        for block in response.content:
            if block.type == "text":
                reasoning_text = block.text
            elif block.type == "tool_use":
                action_name = block.name
                action_input = block.input or {}

        # Record the assistant's response in history
        self._push_message({"role": "assistant", "content": response.content})
        # Add tool result to keep the conversation valid
        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use_block:
            self._push_message(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": "Action submitted. Waiting for next frame...",
                        }
                    ],
                }
            )

        logger.info(f"Claude chose: {action_name} with input {action_input}")

        action = GameAction.from_name(action_name)
        if action_input:
            action.set_data(action_input)
        action.reasoning = {
            "model": self.MODEL,
            "observation": reasoning_text[:500] if reasoning_text else "",
            "action_chosen": action_name,
            "total_tokens": self.token_counter,
        }
        return action

    def _track_tokens(self, tokens: int) -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {"tokens": tokens, "total_tokens": self.token_counter}
            )
        logger.info(f"Tokens used: {tokens}, total: {self.token_counter}")

    def _push_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            # Keep the system context (first message) and trim old messages
            self.messages = [self.messages[0]] + self.messages[-(self.MESSAGE_LIMIT - 1) :]
            # Ensure we don't start with a tool_result
            while len(self.messages) > 1:
                content = self.messages[1].get("content", "")
                if isinstance(content, list) and any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in content
                ):
                    self.messages.pop(1)
                else:
                    break

    def _build_system_context(self) -> str:
        return textwrap.dedent("""\
            # ARC-AGI-3 Game Agent

            You are an intelligent agent playing ARC-AGI-3, a dynamic grid-based game.
            Your objective is to WIN while minimizing the number of actions taken.

            ## How the game works:
            - You see a 64x64 grid where each cell contains a value 0-15 (representing colors)
            - Each action you take produces a new frame (game state)
            - A frame can contain one or more grids
            - The game has multiple levels; complete all levels to WIN
            - If you run out of energy or make wrong moves, you get GAME_OVER
            - After GAME_OVER, you can RESET to try again

            ## Available actions:
            - RESET: Start or restart the game
            - ACTION1 (Up/W): Move up
            - ACTION2 (Down/S): Move down
            - ACTION3 (Left/A): Move left
            - ACTION4 (Right/D): Move right
            - ACTION5 (Enter/Space): Confirm/interact
            - ACTION6 (Click/Point): Click at x,y coordinates (0-63)

            ## Strategy tips:
            - Carefully observe how the grid changes after each action
            - Look for patterns: player position, walls, doors, keys, items
            - Track what works and what doesn't
            - Walls (often color 10) block movement
            - Floors (often color 8) are walkable
            - Look for interactive objects and figure out what they do
            - If the grid doesn't change after a move, you probably hit a wall

            Play strategically and learn from each observation!""")

    def _build_observation(self, frame: FrameData) -> str:
        grid_str = self._pretty_print_grid(frame.frame)
        return textwrap.dedent(f"""\
            # Current Game State

            **State:** {frame.state.name}
            **Levels Completed:** {frame.levels_completed}
            **Win Levels:** {frame.win_levels}
            **Action Count:** {self.action_counter}

            ## Grid:
            {grid_str}

            Analyze this frame carefully. What do you observe? Where is the player?
            What objects or patterns do you see? What should you do next?""")

    def _pretty_print_grid(self, frame: list[list[list[Any]]]) -> str:
        lines = []
        for i, grid in enumerate(frame):
            lines.append(f"Grid {i}:")
            for row in grid:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def _build_tools(self) -> list[dict[str, Any]]:
        empty_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        return [
            {
                "name": "RESET",
                "description": "Start or restart the game. Call this when the game state is NOT_PLAYED or GAME_OVER.",
                "input_schema": empty_schema,
            },
            {
                "name": "ACTION1",
                "description": "Move UP (W key). Use to move the player upward on the grid.",
                "input_schema": empty_schema,
            },
            {
                "name": "ACTION2",
                "description": "Move DOWN (S key). Use to move the player downward on the grid.",
                "input_schema": empty_schema,
            },
            {
                "name": "ACTION3",
                "description": "Move LEFT (A key). Use to move the player to the left on the grid.",
                "input_schema": empty_schema,
            },
            {
                "name": "ACTION4",
                "description": "Move RIGHT (D key). Use to move the player to the right on the grid.",
                "input_schema": empty_schema,
            },
            {
                "name": "ACTION5",
                "description": "CONFIRM / INTERACT (Enter/Space). Use to interact with objects or confirm selections.",
                "input_schema": empty_schema,
            },
            {
                "name": "ACTION6",
                "description": "CLICK at specific coordinates. Use to interact with a specific grid cell.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "integer",
                            "description": "X coordinate (0-63)",
                        },
                        "y": {
                            "type": "integer",
                            "description": "Y coordinate (0-63)",
                        },
                    },
                    "required": ["x", "y"],
                },
            },
        ]

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "claude_model": self.MODEL,
                    "total_tokens": self.token_counter,
                    "total_messages": len(self.messages),
                    "claude_tools": self._build_tools(),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ClaudeFast(ClaudeAgent):
    """Claude agent without observation step - faster but less strategic."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = "claude-haiku-4-5-20251001"


class ClaudeOpus(ClaudeAgent):
    """Claude Opus agent - maximum reasoning power."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL = "claude-opus-4-20250514"
