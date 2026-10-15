from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .game import check_winner, legal_moves, make_move, render_board


def normalize_state(board: np.ndarray, perspective: int) -> Tuple[int, ...]:
    arr = np.asarray(board, dtype=np.int8).copy()
    if perspective == -1:
        arr = -arr
    return tuple(arr.tolist())


class RandomAgent:
    def act(self, board: np.ndarray, player: int) -> int:
        legal = legal_moves(board)
        return random.choice(legal) if legal else -1


class RuleBasedAgent:
    def act(self, board: np.ndarray, player: int) -> int:
        legal = legal_moves(board)
        if not legal:
            return -1

        for move in legal:
            trial = make_move(board, move, player)
            if check_winner(trial) == player:
                return move

        opponent = -player
        for move in legal:
            trial = make_move(board, move, opponent)
            if check_winner(trial) == opponent:
                return move

        preferred = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        for move in preferred:
            if move in legal:
                return move
        return random.choice(legal)


class QLearningAgent:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple[int, ...], Dict[int, float]] = {}

    def _state_key(self, board: np.ndarray, player: int) -> Tuple[int, ...]:
        return normalize_state(board, player)

    def act(self, board: np.ndarray, player: int) -> int:
        legal = legal_moves(board)
        if not legal:
            return -1

        state = self._state_key(board, player)
        if random.random() < self.epsilon:
            return random.choice(legal)

        q_values = self.q_table.setdefault(state, {a: 0.0 for a in range(9)})
        best_value = max(q_values.get(action, 0.0) for action in legal)
        best_moves = [action for action in legal if abs(q_values.get(action, 0.0) - best_value) < 1e-9]
        return random.choice(best_moves)

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...], done: bool) -> None:
        current_q = self.q_table.setdefault(state, {a: 0.0 for a in range(9)})
        current_value = current_q.get(action, 0.0)

        if done:
            target = reward
        else:
            next_q = self.q_table.setdefault(next_state, {a: 0.0 for a in range(9)})
            target = reward + self.gamma * max(next_q.values())

        current_q[action] = current_value + self.alpha * (target - current_value)
        self.q_table[state] = current_q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, path: str) -> None:
        import pickle

        with open(path, "rb") as handle:
            self.q_table = pickle.load(handle)

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as handle:
            pickle.dump(self.q_table, handle)


class DQN(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, output_dim: int = 9):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995, gamma: float = 0.9, batch_size: int = 64, buffer_size: int = 20000):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def _state_array(self, board: np.ndarray, player: int) -> np.ndarray:
        arr = np.asarray(board, dtype=np.int8).copy()
        if player == -1:
            arr = -arr
        return arr.astype(np.float32)

    def act(self, board: np.ndarray, player: int) -> int:
        legal = legal_moves(board)
        if not legal:
            return -1

        if random.random() < self.epsilon:
            return random.choice(legal)

        state = torch.tensor([self._state_array(board, player)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(state)[0].cpu().numpy()

        for invalid_action in set(range(9)) - set(legal):
            q_values[invalid_action] = -1e9

        return int(np.argmax(q_values))

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.remember(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device)

        current_q = self.model(states)
        next_q = self.target_model(next_states).detach()
        target_q = current_q.clone()

        max_next = torch.max(next_q, dim=1).values
        target_q[torch.arange(self.batch_size, device=self.device), actions] = rewards + (1.0 - dones) * self.gamma * max_next

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path: str) -> None:
        torch.save({"model_state_dict": self.model.state_dict(), "target_state_dict": self.target_model.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)


def play_game(agent, opponent, agent_symbol: int = 1, verbose: bool = False):
    from .env import TicTacToeEnv

    env = TicTacToeEnv()
    agent_moves = []

    while not env.done:
        if env.current_player == agent_symbol:
            state = normalize_state(env.board, agent_symbol)
            action = agent.act(env.board, env.current_player)
            if action == -1:
                break
            next_board, _, done, _, _ = env.step(action)
            next_state = normalize_state(next_board, agent_symbol)
            agent_moves.append((state, action, next_state, done))
            if verbose:
                print(f"Agent plays {action} on board:\n{render_board(env.board)}")
            if done:
                break
        else:
            opponent_action = opponent.act(env.board, env.current_player)
            if opponent_action == -1:
                break
            _, _, done, _, _ = env.step(opponent_action)
            if verbose:
                print(f"Opponent plays {opponent_action} on board:\n{render_board(env.board)}")
            if done:
                break

    outcome = env.winner
    if outcome == 0:
        outcome_score = 0.0
    elif outcome == agent_symbol:
        outcome_score = 1.0
    else:
        outcome_score = -1.0

    return outcome_score, agent_moves
