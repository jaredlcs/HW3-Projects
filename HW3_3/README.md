# HW3-3: Selfmade DodgerEnv with PPO

## Demo Video

**[Watch the trained agent in action on YouTube](https://youtu.be/OyhIjeMFz6o)** 🎮

## Implementation Details

This project implements PPO **from scratch** without using high-level RL libraries

## Requirements

- Python 3.8 or higher
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

**Not used:** Stable-Baselines3, RLlib, or any RL frameworks

Install everything with:

```bash
pip install -r requirements.txt
```

## Hyperparameters

These are the settings we used to train the agent:

| Parameter | Value | What It Does |
|-----------|-------|--------------|
| Learning Rate | 0.0003 | How fast the agent learns |
| Gamma (γ) | 0.99 | How much to value future rewards |
| GAE Lambda (λ) | 0.95 | Balance between bias and variance |
| Clip Epsilon (ε) | 0.2 | Prevents too-large policy updates |
| T Horizon | 4096 | Steps collected before each update |
| K Epochs | 10 | Training passes per batch |
| Batch Size | 64 | Mini-batch size for training |
| Entropy Coef | 0.01 | Encourages exploration |
| Value Coef | 0.5 | Weight for value function loss |
| Hidden Dim | 128 | Neural network size |
| Max Episodes | 3000 | Maximum training episodes |
| Target Reward | 1200 | Success threshold (100-ep average) |

## Quick Start

You can run this project **locally** on your machine.

### Run Locally

```bash
pip install -r requirements.txt
python main.py
```

## What This Does

This project trains an AI agent to dodge falling obstacles using **PPO (Proximal Policy Optimization)**, a state-of-the-art reinforcement learning algorithm. The agent learns by trial and error to move left or right to avoid being hit by falling obstacles.

**Goal:** Achieve an average reward ≥ 1200 over 100 episodes (indicating consistent successful dodges)

## How It Works

### The Environment

The agent controls a dodger that must avoid a falling obstacle:

**Actions:**
- **Do nothing** – Stay in place
- **Move left** – Shift dodger left
- **Move right** – Shift dodger right

**Observations:**
- Dodger's horizontal position (normalized)
- Obstacle's horizontal position (normalized)
- Obstacle's vertical position (normalized)
- Obstacle's vertical speed (normalized)

**Rewards:**
- +1 point per frame survived
- Small bonus for distance from obstacle
- Collision: -100 points (episode ends)

### The Algorithm: PPO

PPO (Proximal Policy Optimization) uses two neural networks:
- **Actor:** Decides which action to take
- **Critic:** Evaluates how good the current state is

The algorithm learns by:
1. Collecting experience from the environment
2. Computing how much better/worse actions were than expected
3. Updating the policy to favor better actions
4. Repeating until the agent masters dodging
## How to Use

### Train a New Agent

```bash
python main.py
```

### Resume Training (if interrupted)

```bash
python main.py --resume
```

### Test the Trained Agent

```bash
python main.py --test
```

For more episodes:

```bash
python main.py --test --test-episodes 20
```

## What Gets Saved

After training, you'll see these files:

- **`model.pth`** - The trained agent (best model)
- **`train_plot.png`** - Graph showing learning progress
- **`checkpoints/`** - Saved progress (for resuming training)
- various metadata and pkl files which store progress for checkpoints/plotting



## References
- **Course:** CSCI6353 Reinforcement Learning

---

**Author:** Jared Soto  
**Assignment:** CSCI6353 Homework 3, Part 3
**Date:** November 2025
