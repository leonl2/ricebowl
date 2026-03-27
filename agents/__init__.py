from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .competition.graph_cnn_agent import GraphCNNAgent
from .recorder import Recorder
from .swarm import Swarm
from .templates.claude_agent import ClaudeAgent, ClaudeFast, ClaudeOpus
from .templates.explorer_agent import Explorer
from .templates.hybrid_agent import HybridAgent
from .templates.local_llm_agent import LocalLLMAgent, LocalLLMFast
from .templates.random_agent import Random

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# Register all agent variants explicitly
AVAILABLE_AGENTS["random"] = Random
AVAILABLE_AGENTS["explorer"] = Explorer
AVAILABLE_AGENTS["local"] = LocalLLMAgent
AVAILABLE_AGENTS["localfast"] = LocalLLMFast
AVAILABLE_AGENTS["hybrid"] = HybridAgent
AVAILABLE_AGENTS["claude"] = ClaudeAgent
AVAILABLE_AGENTS["claudefast"] = ClaudeFast
AVAILABLE_AGENTS["claudeopus"] = ClaudeOpus
AVAILABLE_AGENTS["graphcnn"] = GraphCNNAgent  # Competition agent

# Add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

__all__ = [
    "Swarm",
    "Random",
    "Explorer",
    "LocalLLMAgent",
    "LocalLLMFast",
    "HybridAgent",
    "ClaudeAgent",
    "ClaudeFast",
    "ClaudeOpus",
    "GraphCNNAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
