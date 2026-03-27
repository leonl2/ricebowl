#!/usr/bin/env python3
"""Entry point for ARC-AGI-3 agents."""

import argparse
import json
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

# Set up logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", mode="a"),
    ],
)
logger = logging.getLogger()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Agent Runner")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent to use (e.g., random, claude, claudefast, claudeopus)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Specific game ID to play (e.g., ls20). If not set, plays all available.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=[],
        help="Tags for the scorecard",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Run in offline mode (no ARC API key needed, no scorecards)",
    )
    args = parser.parse_args()

    # Set operation mode env var before importing agents
    if args.offline:
        os.environ["ONLINE_ONLY"] = "False"
        os.environ["ARC_OPERATION_MODE"] = "OFFLINE"

    ROOT_URL = os.getenv("ARC_BASE_URL", "https://three.arcprize.org")

    from agents import AVAILABLE_AGENTS, Swarm

    agent_name = args.agent.lower()
    if agent_name not in AVAILABLE_AGENTS:
        print(f"Unknown agent: {agent_name}")
        print(f"Available agents: {', '.join(sorted(AVAILABLE_AGENTS.keys()))}")
        sys.exit(1)

    # Determine games to play
    if args.game:
        games = [args.game]
    else:
        # Default to ls20 if no game specified
        games = ["ls20"]

    mode = "OFFLINE" if args.offline else "ONLINE"
    print(f"Starting agent '{agent_name}' on games: {games} (mode: {mode})")
    print(f"API URL: {ROOT_URL}")

    swarm = Swarm(
        agent=agent_name,
        ROOT_URL=ROOT_URL,
        games=games,
        tags=args.tags,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig: int, frame: object) -> None:
        print("\nInterrupted! Cleaning up...")
        if swarm.card_id:
            scorecard = swarm.close_scorecard(swarm.card_id)
            if scorecard:
                print("--- FINAL SCORECARD ---")
                print(json.dumps(scorecard.model_dump(), indent=2))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    scorecard = swarm.main()

    if scorecard:
        print("\n=== FINAL RESULTS ===")
        print(json.dumps(scorecard.model_dump(), indent=2))
    else:
        print("\nNo scorecard returned.")


if __name__ == "__main__":
    main()
