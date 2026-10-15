# Appendix A: Major Project Proposal

## Title
Reinforcement Learning-Based Intelligent Game Agent

## Student Details
- Student Name: Sahildeep Sharma
- Roll No.: AA.SC.P2MCA25010057
- Modality: Independent Exploration (continuation of the minor project)
- Address of the Company: Not applicable (Independent Exploration)
- Guide: Not applicable (Industry Integration not chosen)

## Abstract
Games are a well-established testbed for Artificial Intelligence because they offer clear rules, measurable outcomes and a safe space for experimentation. In the minor project, an AI-based game was developed whose opponent relied on fixed rules or search-based logic. This major project extends that work by building an intelligent game agent that learns to play through Reinforcement Learning (RL) instead of following hand-written rules.

The current project wraps the game as a Gymnasium-compatible RL environment with a carefully designed state representation, action space and reward function. Two learning agents were implemented and trained: a tabular Q-learning agent and a Deep Q-Network (DQN) agent using experience replay and a target network. Their performance was compared against a random agent and a rule-based baseline using win rate, average reward and episode outcomes. The resulting implementation is a working project with training scripts, evaluation utilities and a human-vs-agent demo.

## Assumptions and Declarations
- The underlying game logic can be separated from its visual interface and run headlessly during training.
- The game state can be represented numerically and the action space is finite and discrete.
- Model training is designed to run on a standard laptop CPU or free notebook environment without paid compute.
- The code, experiments, analysis and conclusions are the author’s own work, and AI tools are used only for assistance and learning within the permitted limits.
- All referenced libraries and technologies are cited and the project documentation is kept reproducible and traceable.

## Main Objective
To design, train and evaluate Reinforcement Learning agents that learn to play the game autonomously and compare them with baseline opponent strategies.

## Deliverables
- Gymnasium-compatible environment built around the game logic.
- Random and rule-based baseline agents.
- Q-learning and DQN agent implementations with saved trained models.
- Evaluation results and learning-curve style metrics.
- A human-vs-agent mode for demonstration.
- Final project source code and documentation.

## Timeline and Milestones
1. Proposal and title selection.
2. Literature review and environment design.
3. Environment wrapper and baseline agents.
4. Interim report and design documentation.
5. Q-learning and DQN training.
6. Evaluation and hyperparameter tuning.
7. Demo preparation and code cleanup.
8. Final report and submission.
9. Final review and viva.

## Tools
- Python 3.10+
- Gymnasium
- PyTorch
- NumPy, Pandas, Matplotlib
- Pygame (where needed for UI or demo)
- Git/GitHub and TensorBoard-style logging
- Standard laptop / free cloud notebook for training

## Learning Outcomes
- Reinforcement Learning fundamentals, including MDPs and value-based learning.
- Q-learning and DQN architecture and training behaviour.
- Reward design and environment modelling.
- Evaluation, tuning and reproducible experimentation.
- Software engineering for a complete AI application.

## Project Implementation Status
This repository contains the working implementation for the proposal: a complete Tic-Tac-Toe RL project that includes a Gymnasium environment, Q-learning and DQN training, evaluation scripts and human-versus-agent play. The project is ready to train, compare and demonstrate within the workspace.
