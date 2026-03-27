# ricebowl

ARC-AGI-3 agent powered by Claude.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Copy `.env.example` to `.env` and fill in your API keys:
   - `ANTHROPIC_API_KEY` - your Anthropic API key (required for Claude agents)
   - `ARC_API_KEY` - get one free at https://three.arcprize.org (optional, for online scorecards)
3. Run an agent:

```bash
# Run locally (no ARC API key needed, no scorecards)
uv run main.py --agent=claude --game=ls20 --offline

# Run the random baseline (no LLM key needed either)
uv run main.py --agent=random --game=ls20 --offline

# Run online (gets scorecards + replays, needs ARC_API_KEY)
uv run main.py --agent=claude --game=ls20

# Fast Claude agent (Haiku, no observation step)
uv run main.py --agent=claudefast --game=ls20

# Claude Opus (maximum reasoning)
uv run main.py --agent=claudeopus --game=ls20
```

## Costs

- **ARC-AGI-3 game access**: Free. Games run locally via the `arc-agi` toolkit.
- **LLM API calls**: Uses your own Anthropic API key/credits. The `random` agent needs no LLM.
- **Competition submissions**: Run offline on Kaggle with no internet ($50 compute budget).

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
