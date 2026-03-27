from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.claude_agent import ClaudeAgent, ClaudeFast, ClaudeOpus
from .templates.random_agent import Random

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# Register Claude agent variants explicitly
AVAILABLE_AGENTS["claude"] = ClaudeAgent
AVAILABLE_AGENTS["claudefast"] = ClaudeFast
AVAILABLE_AGENTS["claudeopus"] = ClaudeOpus
AVAILABLE_AGENTS["random"] = Random

# Add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

__all__ = [
    "Swarm",
    "Random",
    "ClaudeAgent",
    "ClaudeFast",
    "ClaudeOpus",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
