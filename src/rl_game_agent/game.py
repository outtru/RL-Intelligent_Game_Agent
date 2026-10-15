from __future__ import annotations

from typing import List, Optional

import numpy as np

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def legal_moves(board: np.ndarray) -> List[int]:
    return [idx for idx, value in enumerate(board) if value == 0]


def check_winner(board: np.ndarray) -> Optional[int]:
    for line in WIN_LINES:
        first = board[line[0]]
        if first != 0 and all(board[index] == first for index in line[1:]):
            return int(first)

    if np.all(board != 0):
        return 0

    return None


def make_move(board: np.ndarray, action: int, player: int) -> np.ndarray:
    next_board = board.copy()
    if not 0 <= action < len(next_board):
        raise ValueError(f"Action {action} is outside the valid range 0..{len(next_board) - 1}.")
    if next_board[action] != 0:
        raise ValueError(f"Action {action} is already occupied.")
    next_board[action] = player
    return next_board


def render_board(board: np.ndarray) -> str:
    symbols = {1: "X", -1: "O", 0: "-"}
    rows = []
    for row in range(3):
        row_values = []
        for col in range(3):
            idx = row * 3 + col
            row_values.append(symbols[int(board[idx])])
        rows.append(" | ".join(row_values))
    return "\n".join(rows)
