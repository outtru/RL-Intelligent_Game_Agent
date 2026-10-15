# Reinforcement Learning-Based Intelligent Game Agent

This project implements a full RL-based game agent for Tic-Tac-Toe and packages it as a Gymnasium-compatible environment with:

- a tabular Q-learning agent
- a Deep Q-Network (DQN) agent
- a random baseline
- a rule-based baseline using simple look-ahead logic
- evaluation scripts and a human-vs-agent demo

It is designed to match the structure and goals described in the major-project proposal: a clear RL environment, comparison against baselines, and a working playable agent.

## Project structure

- `src/rl_game_agent/game.py` - core game logic and winner detection
- `src/rl_game_agent/env.py` - Gymnasium environment wrapper
- `src/rl_game_agent/agents.py` - agents and training helpers
- `src/rl_game_agent/train.py` - training scripts for Q-learning and DQN
- `src/rl_game_agent/evaluate.py` - evaluation and metrics
- `src/rl_game_agent/demo.py` - human vs trained agent demo
- `main.py` - command-line entry point

## Quick start

1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Train the agents:
   
   ```bash
   python main.py train --agent q
   python main.py train --agent dqn
   ```
4. Evaluate the trained agent:
   
   ```bash
   python main.py evaluate --agent q
   python main.py evaluate --agent dqn
   ```
5. Play a demo against the trained agent:
   
   ```bash
   python main.py demo --agent q
   ```

## Example usage

```bash
python main.py train --agent q --episodes 500
python main.py evaluate --agent q --episodes 200
python main.py demo --agent dqn
```

## Notes

The game is intentionally minimalist and fast to train, which makes it ideal for a first RL project. The environment and agent APIs are modular, so the same pattern can be extended to a more complex game in the future.
