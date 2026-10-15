from __future__ import annotations

import os

from .agents import DQNAgent, QLearningAgent
from .env import TicTacToeEnv
from .game import render_board


def human_move(board):
    while True:
        try:
            raw = input("Choose a square (0-8): ")
            index = int(raw)
            if index < 0 or index > 8:
                raise ValueError
            if board[index] != 0:
                print("That square is already occupied.")
                continue
            return index
        except ValueError:
            print("Please enter a valid integer between 0 and 8.")


def play_human_vs_agent(agent_name: str = "q"):
    env = TicTacToeEnv()

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

    print("You are X. The trained agent is O.")
    print(render_board(env.board))

    while not env.done:
        if env.current_player == 1:
            action = human_move(env.board)
            _, _, done, _, _ = env.step(action)
        else:
            action = agent.act(env.board, env.current_player)
            print(f"Agent chooses square {action}.")
            _, _, done, _, _ = env.step(action)

        print(render_board(env.board))
        if done:
            break

    if env.winner == 1:
        print("You win!")
    elif env.winner == -1:
        print("The trained agent wins.")
    else:
        print("Draw!")
