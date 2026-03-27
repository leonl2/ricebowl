# ricebowl

ARC-AGI-3 agent powered by Claude.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Copy `.env.example` to `.env` and fill in your API keys:
   - `ARC_API_KEY` - get one at https://three.arcprize.org
   - `ANTHROPIC_API_KEY` - your Anthropic API key
3. Run an agent:

```bash
# Run the Claude agent on the ls20 game
uv run main.py --agent=claude --game=ls20

# Run the fast Claude agent (Haiku, no observation step)
uv run main.py --agent=claudefast --game=ls20

# Run Claude Opus (maximum reasoning)
uv run main.py --agent=claudeopus --game=ls20

# Run the random baseline agent
uv run main.py --agent=random --game=ls20
```

## Agents

| Agent | Model | Description |
|-------|-------|-------------|
| `claude` | Claude Sonnet | Observes state, reasons, then picks actions via tool use |
| `claudefast` | Claude Haiku | Skips observation step for speed |
| `claudeopus` | Claude Opus | Maximum reasoning power |
| `random` | N/A | Random action baseline |

## How it works

Each agent plays ARC-AGI-3 games - turn-based 2D grid environments where the agent must figure out the rules and objectives by interacting with the environment. There are no instructions given.

The Claude agent:
1. Observes the 64x64 grid state
2. Reasons about what it sees (player position, objects, patterns)
3. Uses tool_use to select an action (move, interact, or click)
4. Learns from how the grid changes after each action
