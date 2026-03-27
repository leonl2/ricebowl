# ricebowl

Competition agent for [ARC Prize 2026](https://arcprize.org/competitions/2026/arc-agi-3) (ARC-AGI-3).

Combines the top two approaches from the developer preview:
- **1st place (12.58%)**: CNN that learns which actions change the game state
- **3rd place (~similar post-bugfix)**: Systematic graph-based state exploration

## Quick Start

```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
cp .env.example .env  # Add your ARC_API_KEY

# Run the competition agent
uv run main.py --agent=graphcnn --game=ls20 --offline
```

## The Competition Agent (`graphcnn`)

This is the agent designed for the ARC Prize 2026 leaderboard.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Game Environment (64x64 grid, 16 colors)       │
│  Actions: UP/DOWN/LEFT/RIGHT/INTERACT/CLICK     │
└──────────────────────┬──────────────────────────┘
                       │ frame
          ┌────────────▼────────────┐
          │  Frame Analyzer         │
          │  - Connected components │
          │  - Player detection     │
          │  - Priority scoring     │
          └────┬──────────┬─────────┘
               │          │
    ┌──────────▼──┐  ┌────▼──────────────┐
    │ State Graph │  │ CNN Predictor      │
    │ - Hash      │  │ - 4-layer CNN      │
    │   states    │  │ - Predicts: will   │
    │ - Track     │  │   this action      │
    │   frontier  │  │   change state?    │
    │ - Shortest  │  │ - Spatial heatmap  │
    │   paths     │  │   for click targets│
    │ - Loop      │  │ - Trains online    │
    │   detection │  │   from experience  │
    └──────┬──────┘  └────────┬───────────┘
           │                  │
      ┌────▼──────────────────▼────┐
      │  Action Selection          │
      │  1. Replay queue (nav)     │
      │  2. ε-random exploration   │
      │  3. CNN-guided untested    │
      │  4. Navigate to frontier   │
      │  5. CNN exploitation       │
      └───────────────────────────┘
```

### How It Works

1. **State Graph**: Every frame is hashed into a compact state ID. Actions create edges.
   The graph tracks which actions have been tried at each state, detects loops, and finds
   shortest paths to "frontier" states (states with untested actions).

2. **CNN Action Predictor**: A 4-layer CNN (input: 16-channel one-hot 64x64 grid) predicts
   whether each action will change the game state. Trained online from an experience buffer
   of ~200K deduplicated (state, action, did_change) tuples. Two output heads: sigmoid probs
   for ACTION1-5, and a 64x64 spatial heatmap for ACTION6 click targets.

3. **Action Selection**: Prioritizes untested actions at the current state (CNN picks the most
   promising). When a state is fully explored, navigates to the nearest frontier state via
   shortest path. Falls back to component-priority analysis when CNN hasn't trained enough.

4. **Adaptation**: CNN resets weights on level changes. Player position tracked via movement
   diff detection. Status bars masked from state hashing to avoid state explosion.

### Why Not LLMs?

Frontier LLMs score **under 1%** on ARC-AGI-3 (GPT-5.4: 0.26%, Claude Opus: 0.25%).
The 1st place CNN scored 12.58%. LLMs fail because:
- They can't do systematic state-space exploration
- They waste actions (RHAE scoring penalizes quadratically: 10x human actions = 1% credit)
- They can't build world models from interaction
- Duke testing: Claude Opus scores 97% with hand-crafted scaffolding, 0% on unfamiliar games

## All Agents

| Agent | Type | Description |
|-------|------|-------------|
| **`graphcnn`** | Competition | Graph search + CNN (best agent) |
| `explorer` | Programmatic | Grid analysis + systematic exploration (no ML) |
| `hybrid` | Local LLM | Grid analysis feeds local LLM via OpenAI-compatible API |
| `local` | Local LLM | Direct local LLM with tool calling |
| `claude` | API | Claude Sonnet via Anthropic API |
| `random` | Baseline | Random actions |

## Local Model Setup (for `hybrid`/`local` agents)

```bash
ollama pull qwen2.5:32b-instruct && ollama serve
# Then: uv run main.py --agent=hybrid --game=ls20 --offline
```

## Scoring (RHAE)

ARC-AGI-3 uses Relative Human Action Efficiency:
- Score = (human_actions / AI_actions)^2
- Hard cutoff at 5x human actions
- 10x more actions than human = only 1% credit
- This is why systematic exploration beats brute-force LLM reasoning
