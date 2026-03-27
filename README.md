# ricebowl

Competition-ready ARC-AGI-3 agents for [ARC Prize 2026](https://arcprize.org/competitions/2026).

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Copy `.env.example` to `.env` and fill in your keys
3. Run an agent:

```bash
# === COMPETITION-READY (no internet needed) ===

# Programmatic explorer - no LLM needed at all
uv run main.py --agent=explorer --game=ls20 --offline

# Local LLM agent - requires local model server (ollama/vLLM/SGLang)
uv run main.py --agent=local --game=ls20 --offline

# Hybrid agent - grid analysis + local LLM (best performance)
uv run main.py --agent=hybrid --game=ls20 --offline

# === DEVELOPMENT (uses API credits) ===

# Claude Sonnet - observe + reason + act
uv run main.py --agent=claude --game=ls20

# Random baseline
uv run main.py --agent=random --game=ls20 --offline
```

## Agents

### Competition-Ready (no internet required)

| Agent | Needs LLM? | Description |
|-------|-----------|-------------|
| `explorer` | No | Programmatic grid analysis, change detection, systematic exploration |
| `local` | Local model | OpenAI-compatible API (ollama/vLLM/SGLang) with tool calling |
| `localfast` | Local model | Same as `local` but skips observation step |
| `hybrid` | Local model | **Best agent** - programmatic analysis feeds compact summaries to local LLM |

### Development (API credits)

| Agent | Model | Description |
|-------|-------|-------------|
| `claude` | Claude Sonnet | Full observe-reason-act loop via Anthropic API |
| `claudefast` | Claude Haiku | Fast, skips observation |
| `claudeopus` | Claude Opus | Maximum reasoning power |
| `random` | N/A | Random baseline |

## Local Model Setup

For competition agents, you need a local model server. Recommended setup:

```bash
# Option 1: Ollama (easiest)
ollama pull qwen2.5:32b-instruct
ollama serve

# Option 2: SGLang (fastest - has prefix caching)
pip install sglang
python -m sglang.launch_server --model Qwen/Qwen2.5-32B-Instruct --port 8000

# Option 3: vLLM
pip install vllm
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000
```

Configure in `.env`:
```
LOCAL_LLM_BASE_URL=http://localhost:11434/v1   # ollama default
LOCAL_LLM_MODEL=qwen2.5:32b-instruct
LOCAL_LLM_API_KEY=ollama
```

**Recommended model**: Qwen2.5-32B-Instruct fits in 32GB VRAM at Q5 quantization
with ~40-60 tok/s on RTX 5090. Strong tool calling and spatial reasoning.

## Architecture

```
agents/
├── agent.py              # Base Agent + Playback
├── swarm.py              # Multi-game orchestration
├── recorder.py           # Session recording (JSONL)
├── tracing.py            # AgentOps integration
└── templates/
    ├── explorer_agent.py  # Programmatic: grid analysis + systematic exploration
    ├── hybrid_agent.py    # Hybrid: grid analysis → compact summary → local LLM
    ├── local_llm_agent.py # Local LLM: OpenAI-compatible API with tool use
    ├── claude_agent.py    # Claude API agent (development)
    └── random_agent.py    # Random baseline
```

### How the Hybrid Agent Works

1. **GridAnalyzer** extracts structured features from the 64x64 grid:
   - Player position detection via movement diff analysis
   - Object detection (connected components by color)
   - Wall mapping from failed movement attempts
   - Status row parsing (energy, score, level info)

2. **Compact summary** sent to the local LLM instead of raw grid data:
   - "3 cells changed, player moved to (25,30), 4 objects visible, nearest is color=6 at dist=8"
   - Dramatically reduces tokens vs. sending all 4096 cells

3. **Local LLM** receives pre-digested analysis and picks actions via tool calling

4. **Cross-death learning**: action history and wall maps persist across GAME_OVERs

## Costs

| Component | Cost |
|-----------|------|
| ARC-AGI-3 games | Free (runs locally) |
| `explorer` agent | Free (no LLM) |
| `local`/`hybrid` agents | Free (your own GPU) |
| `claude` agents | Your Anthropic API credits |
| Competition submissions | RTX 5090, 8 hours, no internet |
