# Frozen Lake 

This repository demonstrates the implementation and comparison of three Reinforcement Learning (RL) algorithms on the classic Frozen Lake environment from OpenAI Gymnasium:

- *Q-Learning*
- *SARSA*
- *Deep Q-Network (DQN)*

## Table of Contents

- [Overview](#overview)
- [Environment](#environment)
- [Implemented Algorithms](#implemented-algorithms)
  - [Q-Learning](#1-q-learning)
  - [SARSA](#2-sarsa)
  - [Deep Q-Network (DQN)](#3-deep-q-network-dqn)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Notebook](#running-the-notebook)
- [Results and Visualizations](#results-and-visualizations)
- [File Structure](#file-structure)
- [References](#references)

---

## Overview

This project explores model-free RL algorithms to solve the stochastic Frozen Lake problem. The task is to train agents to navigate a slippery 4x4 grid to reach the goal while avoiding holes.

The notebook RL_PROJECT.ipynb contains implementations for:
- *Tabular Q-Learning*
- *Tabular SARSA*
- *Deep Q-Network (DQN) with experience replay*

Evaluation after training shows the agent's success rate over 1000 test episodes.

---

## Environment

- *FrozenLake-v1* from [OpenAI Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- *Map*: 4x4 grid
- *States*: 16 (each cell in the grid)
- *Actions*: 4 (left, down, right, up)
- *Stochastic transitions*: The agent can slip and end up in a different direction.

---

## Implemented Algorithms

### 1. Q-Learning

- *Off-policy* TD control.
- Updates Q-values using the maximum future Q-value for the next state.
- Includes epsilon-greedy exploration and decaying learning rate.

#### Highlights
- Stores Q-table as a NumPy array.
- Rewards are shaped to speed up learning (higher positive for reaching the goal, negative for falling into a hole).

### 2. SARSA

- *On-policy* TD control.
- Updates Q-values using the Q-value of the actual action taken in the next state.
- Epsilon-greedy action selection.

#### Highlights
- On-policy: updates according to actions taken by the current policy.
- Visualizes the learned policy with arrows and value annotations.

### 3. Deep Q-Network (DQN)

- Uses a neural network to approximate Q-values.
- Employs experience replay buffer and target network for stability.
- One-hot encodes the discrete states for NN input.

#### Highlights
- Implemented in PyTorch.
- Experience replay for decorrelating updates.
- Target network is periodically updated.
- Value function visualization after training.

---

## Getting Started

### Prerequisites

- Python 3.7+
- Recommended: [Google Colab](https://colab.research.google.com/) (for easier setup with GPU/CPU)

#### Required Libraries

Install dependencies using pip:
bash
pip install gymnasium matplotlib torch numpy


### Running the Notebook

1. Clone the repository:
   bash
   git clone https://github.com/Ritupriya17/Frozen-Lake.git
   cd Frozen-Lake
   

2. Open RL_PROJECT.ipynb in Jupyter Notebook or upload to [Google Colab](https://colab.research.google.com/).

3. Run all cells to train and evaluate each agent.

---

## Results and Visualizations

- *Success Rate*: Plots of agent's success rate over training episodes.
- *Value Function*: Heatmaps or tables of learned value functions.
- *Policy Visualization*: Arrows on the grid showing optimal actions.
- *Training/Evaluation Outputs*: Printed to stdout.

Sample results (may vary by run due to stochasticity):

| Algorithm | Final Success Rate (1000 eval episodes) |
|-----------|----------------------------------------|
| Q-Learning| ~69%                                   |
| SARSA     | ~74%                                   |
| DQN       | ~79%                                   |

Plots and images are saved as PNG files in the working directory.

---

## File Structure


RL_PROJECT.ipynb         # Main Jupyter notebook with all implementations
README.md                # This file
frozen_lake4x4.pkl       # Saved Q-table for Q-Learning
frozen_lake4x4_sarsa.pkl # Saved Q-table for SARSA
frozen_lake_dqn.pth      # Saved PyTorch model for DQN
*.png                    # Visualizations (value function, policy, performance)


---

## References

- [OpenAI Gymnasium - FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

---

## Acknowledgements

- The project uses open-source libraries: Gymnasium, NumPy, PyTorch, and Matplotlib.
- Inspired by classical RL examples and educational materials.
