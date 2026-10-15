from __future__ import annotations

import argparse

from src.rl_game_agent.demo import play_human_vs_agent
from src.rl_game_agent.evaluate import benchmark_agents
from src.rl_game_agent.train import train_agent


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and play with a Tic-Tac-Toe RL agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Q or DQN agent.")
    train_parser.add_argument("--agent", choices=["q", "dqn"], default="q")
    train_parser.add_argument("--episodes", type=int, default=500)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved agent against a baseline.")
    eval_parser.add_argument("--agent", choices=["q", "dqn"], default="q")
    eval_parser.add_argument("--episodes", type=int, default=100)

    demo_parser = subparsers.add_parser("demo", help="Play a human-vs-agent demo.")
    demo_parser.add_argument("--agent", choices=["q", "dqn"], default="q")

    args = parser.parse_args()

    if args.command == "train":
        train_agent(agent_name=args.agent, episodes=args.episodes)
        print(f"Training completed for {args.agent} agent.")
    elif args.command == "evaluate":
        results = benchmark_agents(agent_name=args.agent, episodes=args.episodes)
        print(results)
    elif args.command == "demo":
        play_human_vs_agent(agent_name=args.agent)


if __name__ == "__main__":
    main()
