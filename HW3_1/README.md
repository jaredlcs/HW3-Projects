# HW3-1: LunarLander-v3 with PPO

## Demo Video

**[Watch the trained agent in action on YouTube](https://youtu.be/mQDf61ym2ro)** 🚀

## Implementation Details

This project implements PPO **from scratch** without using high-level RL libraries:

**Libraries used:**
- PyTorch (for neural networks)
- Gymnasium (for the environment only)
- Standard Python libraries (NumPy, Matplotlib)

**Not used:** Stable-Baselines3, RLlib, or any RL frameworks

## Quick Start

You can run this project either on **Google Colab** or **locally** on your machine.

### Option 1: Run on Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaredlcs/HW3-Projects/blob/main/HW3_1/colab_notebook.ipynb)

**Click the "Open in Colab" badge** to open a ready-to-run notebook, or manually:

```python
# Install Box2D dependencies
!apt-get update -qq
!apt-get install -y swig build-essential python3-dev

# Clone and setup
!git clone https://github.com/jaredlcs/HW3-Projects.git
%cd HW3-Projects/HW3_1
!pip install gymnasium[box2d] torch matplotlib

# Train the agent
!python main.py
```

### Option 2: Run Locally

```bash
git clone https://github.com/jaredlcs/HW3-Projects.git
cd HW3-Projects/HW3_1
pip install -r requirements.txt
python main.py
```

## What This Does

This project trains an AI agent to land a lunar module safely using **PPO (Proximal Policy Optimization)**, a state-of-the-art reinforcement learning algorithm. The agent learns by trial and error, controlling thrusters to achieve soft landings.

**Goal:** Achieve an average reward ≥ 200 over 100 episodes (indicating consistent successful landings)

## How It Works

### The Environment

The agent controls a lunar lander with 4 possible actions:
- **Do nothing** - Coast
- **Fire left engine** - Rotate right
- **Fire main engine** - Thrust upward
- **Fire right engine** - Rotate left

The agent observes 8 values: position, velocity, angle, angular velocity, and leg contact states.

**Rewards:**
- Crash: -100 points
- Successful landing: +100 to +200 points
- Fuel usage: Small penalties to encourage efficiency

### The Algorithm: PPO

PPO (Proximal Policy Optimization) uses two neural networks:
- **Actor:** Decides which action to take
- **Critic:** Evaluates how good the current state is

The algorithm learns by:
1. Collecting experience from the environment
2. Computing how much better/worse actions were than expected
3. Updating the policy to favor better actions
4. Repeating until the agent masters landing

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
| Max Episodes | 5000 | Maximum training episodes |
| Target Reward | 200 | Success threshold (100-ep average) |

## Requirements

- Python 3.8 or higher
- PyTorch
- Gymnasium with Box2D
- NumPy
- Matplotlib

Install everything with:

```bash
pip install -r requirements.txt
```

## How to Use

### Train a New Agent

```bash
python main.py
```

This will train the agent from scratch. Training takes about 20-40 minutes on CPU or 10-20 minutes on GPU.

### Resume Training (if interrupted)

```bash
python main.py --resume
```

### Test the Trained Agent

```bash
python main.py --test
```

This runs 10 test episodes with visualization. For more episodes:

```bash
python main.py --test --test-episodes 20
```

## What Gets Saved

After training, you'll see these files:

- **`model.pth`** - The trained agent (best model)
- **`train_plot.png`** - Graph showing learning progress
- **`checkpoints/`** - Saved progress (for resuming training)


## References

- **Paper:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- **Environment:** [Gymnasium LunarLander Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- **Course:** CSCI6353 Reinforcement Learning

---

**Author:** Jared Soto  
**Assignment:** CSCI6353 Homework 3, Part 1  
**Date:** November 2025
