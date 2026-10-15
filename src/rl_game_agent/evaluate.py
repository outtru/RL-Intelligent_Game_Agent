from __future__ import annotations

import os
from typing import Dict

from .agents import DQNAgent, QLearningAgent, RandomAgent, RuleBasedAgent, play_game


def evaluate_agent(agent, opponent, episodes: int = 200, agent_symbol: int = 1) -> Dict[str, float]:
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0.0

    for _ in range(episodes):
        outcome_score, _ = play_game(agent, opponent, agent_symbol=agent_symbol)
        total_reward += outcome_score
        if outcome_score > 0:
            wins += 1
        elif outcome_score < 0:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "average_reward": total_reward / episodes,
        "win_rate": wins / episodes,
    }


def benchmark_agents(agent_name: str = "q", episodes: int = 100) -> Dict[str, Dict[str, float]]:
    if agent_name == "q":
        path = "artifacts/q_agent.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained Q-agent found at {path}. Train it first with 'python main.py train --agent q'.")
        agent = QLearningAgent()
        agent.load(path)
    elif agent_name == "dqn":
        path = "artifacts/dqn_agent.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained DQN agent found at {path}. Train it first with 'python main.py train --agent dqn'.")
        agent = DQNAgent()
        agent.load(path)
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}")

    results = {
        "random": evaluate_agent(agent, RandomAgent(), episodes=episodes),
        "rule_based": evaluate_agent(agent, RuleBasedAgent(), episodes=episodes),
    }
    return results
