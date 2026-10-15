from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from .game import check_winner, legal_moves, make_move, render_board


class TicTacToeEnv(Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, starting_player: int = 1):
        super().__init__()
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = starting_player
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        self.done = False
        self.winner = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            raise ValueError("Cannot take an action after the game has ended.")

        if action not in legal_moves(self.board):
            return self.board.copy(), -1.0, False, False, {"invalid_move": True}

        self.board = make_move(self.board, action, self.current_player)
        winner = check_winner(self.board)

        if winner == self.current_player:
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif winner == 0:
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            self.current_player *= -1
            reward = 0.0
            if not legal_moves(self.board):
                self.done = True
                self.winner = 0
                reward = 0.0

        terminated = self.done
        truncated = False
        return self.board.copy(), float(reward), terminated, truncated, {"winner": self.winner}

    def render(self, mode="human"):
        board = render_board(self.board)
        if mode == "ansi":
            return board
        print(board)

    def close(self):
        pass
