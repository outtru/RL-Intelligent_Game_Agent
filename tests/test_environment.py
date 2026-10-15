import numpy as np

from src.rl_game_agent.env import TicTacToeEnv
from src.rl_game_agent.game import check_winner
from src.rl_game_agent.train import train_q_agent


def test_env_reset_and_step():
    env = TicTacToeEnv()
    observation, info = env.reset()

    assert observation.shape == (9,)
    assert np.all(observation == 0)
    assert env.current_player == 1

    obs, reward, terminated, truncated, info = env.step(0)
    assert obs[0] == 1
    assert terminated is False
    assert reward == 0.0


def test_winner_detection():
    board = np.zeros(9, dtype=np.int8)
    board[[0, 1, 2]] = 1
    assert check_winner(board) == 1


def test_q_training_runs():
    agent = train_q_agent(episodes=20, save_path="/tmp/q_agent_test.pkl")
    assert len(agent.q_table) > 0
