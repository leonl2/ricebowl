"""Local LLM agent using OpenAI-compatible API.

Works with any local inference server that exposes an OpenAI-compatible endpoint:
  - ollama (ollama serve)
  - vLLM (vllm serve)
  - SGLang (python -m sglang.launch_server)
  - llama-server (llama.cpp)

Configure via environment variables:
  LOCAL_LLM_BASE_URL: API endpoint (default: http://localhost:11434/v1 for ollama)
  LOCAL_LLM_MODEL: Model name (default: qwen2.5:32b-instruct)
  LOCAL_LLM_API_KEY: API key if needed (default: ollama)
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

logger = logging.getLogger()


def _safe_parse_action(raw: str) -> Optional[str]:
    """Extract a valid action name from potentially messy LLM output."""
    match = re.search(r"(RESET|ACTION[1-6])", raw)
    return match.group(0) if match else None


class LocalLLMAgent(Agent):
    """Agent powered by a local open-weight LLM via OpenAI-compatible API.

    Designed for competition use - no internet required, runs entirely locally.
    """

    MAX_ACTIONS: int = 80
    MESSAGE_LIMIT: int = 16
    DO_OBSERVATION: bool = True

    messages: list[dict[str, Any]]
    token_counter: int
    _latest_tool_call_id: str = "call_0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0

        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.environ.get("LOCAL_LLM_API_KEY", "ollama")
        self.model = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:32b-instruct")

        self.client = OpenAIClient(base_url=base_url, api_key=api_key)
        self._use_tools = os.environ.get("LOCAL_LLM_USE_TOOLS", "true").lower() == "true"

    @property
    def name(self) -> str:
        sanitized = self.model.replace("/", "-").replace(":", "-")
        return f"{super().name}.local.{sanitized}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First turn: trigger RESET
        if len(self.messages) == 0:
            system_msg = {"role": "system", "content": self._build_system_prompt()}
            self._push_message(system_msg)

            user_msg = {"role": "user", "content": "The game has not started yet. Call RESET to begin."}
            self._push_message(user_msg)

            if self._use_tools:
                assistant_msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {"name": "RESET", "arguments": "{}"},
                        }
                    ],
                }
            else:
                assistant_msg = {"role": "assistant", "content": "RESET"}
            self._push_message(assistant_msg)
            return GameAction.RESET

        # Add the observation from the latest frame
        obs_prompt = self._build_observation(latest_frame)

        if self._use_tools:
            tool_result_msg = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": obs_prompt,
            }
            self._push_message(tool_result_msg)
        else:
            self._push_message({"role": "user", "content": obs_prompt})

        # Optional observation/reasoning step
        if self.DO_OBSERVATION:
            try:
                obs_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    max_tokens=512,
                )
                self._track_tokens(obs_response.usage.total_tokens if obs_response.usage else 0)
                obs_text = obs_response.choices[0].message.content or ""
                logger.info(f"LLM observation: {obs_text[:200]}")
                self._push_message({"role": "assistant", "content": obs_text})
                self._push_message({
                    "role": "user",
                    "content": "Now choose exactly ONE action. Call one action tool.",
                })
            except Exception as e:
                logger.warning(f"Observation step failed: {e}")

        # Action selection
        if self._use_tools:
            return self._choose_with_tools(latest_frame)
        else:
            return self._choose_with_text(latest_frame)

    def _choose_with_tools(self, latest_frame: FrameData) -> GameAction:
        tools = self._build_tools()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools,
                tool_choice="required",
                max_tokens=256,
            )
            self._track_tokens(response.usage.total_tokens if response.usage else 0)

            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                self._latest_tool_call_id = tool_call.id
                action_name = tool_call.function.name
                try:
                    action_input = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    action_input = {}

                self._push_message(message)
                logger.info(f"LLM chose (tool): {action_name}")

                action = GameAction.from_name(action_name)
                if action_input:
                    action.set_data(action_input)
                action.reasoning = {"model": self.model, "action": action_name}
                return action

        except Exception as e:
            logger.warning(f"Tool-based action selection failed: {e}, falling back to text")

        return self._choose_with_text(latest_frame)

    def _choose_with_text(self, latest_frame: FrameData) -> GameAction:
        """Fallback: ask the model to output action name as plain text."""
        if not self._use_tools:
            # Already have context, just need to ask for action
            pass
        else:
            self._push_message({
                "role": "user",
                "content": "Reply with ONLY the action name: RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, or ACTION6.",
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=64,
            )
            self._track_tokens(response.usage.total_tokens if response.usage else 0)
            raw = response.choices[0].message.content or ""
            self._push_message({"role": "assistant", "content": raw})

            action_name = _safe_parse_action(raw)
            if action_name:
                logger.info(f"LLM chose (text): {action_name}")
                action = GameAction.from_name(action_name)
                action.reasoning = {"model": self.model, "raw": raw[:200]}
                return action
        except Exception as e:
            logger.error(f"Text-based action selection failed: {e}")

        # Ultimate fallback
        action = GameAction.ACTION5
        action.reasoning = "fallback"
        return action

    def _track_tokens(self, tokens: int) -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record({"tokens": tokens, "total_tokens": self.token_counter})
        logger.info(f"Tokens: +{tokens} = {self.token_counter}")

    def _push_message(self, message: Any) -> None:
        if isinstance(message, dict):
            self.messages.append(message)
        else:
            self.messages.append(message)

        if len(self.messages) > self.MESSAGE_LIMIT:
            # Keep system prompt + recent messages
            system = [m for m in self.messages[:1] if m.get("role") == "system"]
            rest = self.messages[1:]
            trimmed = rest[-(self.MESSAGE_LIMIT - len(system)):]
            # Don't start with a tool result
            while trimmed and (
                trimmed[0].get("role") == "tool"
                if isinstance(trimmed[0], dict)
                else getattr(trimmed[0], "role", None) == "tool"
            ):
                trimmed.pop(0)
            self.messages = system + trimmed

    def _build_system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are an expert game-playing agent for ARC-AGI-3.
            You play turn-based grid games where you must figure out the rules by experimenting.

            GAME MECHANICS:
            - 64x64 grid, each cell is an integer 0-15 (colors)
            - You take one action per turn and observe the resulting grid
            - Games have multiple levels; complete all to WIN
            - GAME_OVER means you failed; RESET to try again

            ACTIONS:
            - RESET: Start/restart game
            - ACTION1: Up    ACTION2: Down    ACTION3: Left    ACTION4: Right
            - ACTION5: Confirm/interact    ACTION6: Click at (x,y)

            STRATEGY:
            - After each action, compare the new grid to the previous one
            - Identify what changed: your position, objects, counters
            - Walls (color 10) block movement; floors (color 8) are walkable
            - Look for interactive objects (doors, keys, items) by their colors
            - Track energy/score in grid rows 61-63
            - If a move doesn't change the grid, you hit a wall - try a different direction
            - Build a mental map of the environment
            - Be systematic: explore methodically rather than randomly""")

    def _build_observation(self, frame: FrameData) -> str:
        grid_summary = self._summarize_grid(frame.frame)
        return textwrap.dedent(f"""\
            State: {frame.state.name} | Levels: {frame.levels_completed}/{frame.win_levels} | Turn: {self.action_counter}

            {grid_summary}

            What changed? What should you do next? Choose one action.""")

    def _summarize_grid(self, frame: list[list[list[Any]]]) -> str:
        """Create a compact grid summary instead of dumping all 64x64 cells."""
        lines = []
        for i, grid in enumerate(frame):
            lines.append(f"Grid {i} ({len(grid)}x{len(grid[0]) if grid else 0}):")

            # Color frequency analysis
            color_counts: dict[int, int] = {}
            for row in grid:
                for cell in row:
                    color_counts[cell] = color_counts.get(cell, 0) + 1

            # Show color distribution (skip background/dominant color)
            sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
            bg_color = sorted_colors[0][0] if sorted_colors else 0
            notable = [(c, n) for c, n in sorted_colors if c != bg_color and n > 2]
            if notable:
                lines.append(f"  Background: color {bg_color} ({sorted_colors[0][1]} cells)")
                lines.append(f"  Notable: {', '.join(f'color {c}={n} cells' for c, n in notable[:10])}")

            # Show status rows (bottom of grid - often contain score/energy info)
            if len(grid) >= 64:
                lines.append(f"  Row 61 (energy?): {grid[61][:20]}...")
                lines.append(f"  Row 62 (score?):  {grid[62][:20]}...")
                lines.append(f"  Row 63 (info?):   {grid[63][:20]}...")

            # Find non-background regions (potential objects/player)
            objects: list[str] = []
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell != bg_color and cell != 0:
                        # Check if this is a corner of an object cluster
                        if (x == 0 or grid[y][x - 1] == bg_color or grid[y][x - 1] == 0) and \
                           (y == 0 or grid[y - 1][x] == bg_color or grid[y - 1][x] == 0):
                            objects.append(f"({x},{y})=color{cell}")
            if objects:
                lines.append(f"  Object corners (first 20): {', '.join(objects[:20])}")

            # Also show a compact view of the active area
            lines.append(f"  Full grid rows (first 30):")
            for row_idx in range(min(30, len(grid))):
                # Compress row: show only non-bg cells
                row = grid[row_idx]
                non_bg = [(x, v) for x, v in enumerate(row) if v != bg_color]
                if non_bg:
                    lines.append(f"    r{row_idx}: {non_bg}")

            lines.append("")
        return "\n".join(lines)

    def _build_tools(self) -> list[dict[str, Any]]:
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        return [
            {"type": "function", "function": {"name": "RESET", "description": "Start or restart game", "parameters": empty_params, "strict": True}},
            {"type": "function", "function": {"name": "ACTION1", "description": "Move UP", "parameters": empty_params, "strict": True}},
            {"type": "function", "function": {"name": "ACTION2", "description": "Move DOWN", "parameters": empty_params, "strict": True}},
            {"type": "function", "function": {"name": "ACTION3", "description": "Move LEFT", "parameters": empty_params, "strict": True}},
            {"type": "function", "function": {"name": "ACTION4", "description": "Move RIGHT", "parameters": empty_params, "strict": True}},
            {"type": "function", "function": {"name": "ACTION5", "description": "Confirm / interact", "parameters": empty_params, "strict": True}},
            {
                "type": "function",
                "function": {
                    "name": "ACTION6",
                    "description": "Click at coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "string", "description": "X coordinate Int<0,63>"},
                            "y": {"type": "string", "description": "Y coordinate Int<0,63>"},
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
                    "local_model": self.model,
                    "total_tokens": self.token_counter,
                    "total_messages": len(self.messages),
                })
        super().cleanup(*args, **kwargs)


class LocalLLMFast(LocalLLMAgent):
    """Local LLM agent without observation step."""

    DO_OBSERVATION = False
