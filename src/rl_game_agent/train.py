from __future__ import annotations

import os

from .agents import DQNAgent, QLearningAgent, RandomAgent, RuleBasedAgent, play_game


def train_q_agent(episodes: int = 500, save_path: str = "artifacts/q_agent.pkl") -> QLearningAgent:
    agent = QLearningAgent()
    opponent = RandomAgent()

    for episode in range(episodes):
        outcome_score, transitions = play_game(agent, opponent, agent_symbol=1)
        if not transitions:
            continue

        for state, action, next_state, done in transitions:
            agent.learn(state, action, outcome_score, next_state, done)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    agent.save(save_path)
    return agent


def train_dqn_agent(episodes: int = 400, save_path: str = "artifacts/dqn_agent.pt") -> DQNAgent:
    agent = DQNAgent()
    opponent = RuleBasedAgent()

    for episode in range(episodes):
        outcome_score, transitions = play_game(agent, opponent, agent_symbol=1)
        if not transitions:
            continue

        for state, action, next_state, done in transitions:
            agent.learn(state, action, outcome_score, next_state, done)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    agent.save(save_path)
    return agent


def train_agent(agent_name: str = "q", episodes: int = 500):
    if agent_name == "q":
        return train_q_agent(episodes=episodes)
    if agent_name == "dqn":
        return train_dqn_agent(episodes=episodes)
    raise ValueError(f"Unsupported agent name: {agent_name}")
