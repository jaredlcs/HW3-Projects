# HW3-2: Atari Krull-v5 with DQN

## Demo Video
**[Watch the trained agent in action on YouTube](https://youtu.be/ztuCab8BD0o)** 🎮

## Implementation Details

This project implements a **Deep Q-Network (DQN)** from scratch. 

All components—Q-network, replay buffer, preprocessing, training loop, epsilon scheduling, and target network—are written in PyTorch.

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium + ALE Atari environments
- NumPy
- Matplotlib
- SciPy

**Not used:** Stable-Baselines3, RLlib, or any RL frameworks

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Max Steps | 2,000,000 |
| Replay Buffer Size | 300,000 |
| Target Update | Every 10,000 steps |
| Learning Rate | 0.0001 |
| Gamma (γ) | 0.99 |
| Batch Size | 32 |
| Frame Stack | 4 |
| Learning Starts | 50,000 steps |
| Epsilon Start | 1.0 |
| Epsilon End | 0.01 |
| Epsilon Decay Steps | 1,000,000 |

## Quick Start
You can run this project **locally** on your machine.

### Run Locally

```bash
pip install -r requirements.txt
python main.py
```

## What This Does

This project trains a **DQN agent** to play the Atari game **Krull**, learning to:

- Shoot incoming enemies  
- Dodge projectiles  
- Survive chaotic enemy waves  
- Extract spatiotemporal patterns from stacked frames  

## How It Works

### Environment

- **Environment:** `ALE/Krull-v5`  
- **Observation:** (210×160×3) RGB → processed to (84×84) grayscale  
- **Actions:** 18-action Atari movement + FIRE combinations

**Rewards:** 
- Points increase whenever the glaive hits or destroys an enemy (larger foes yield higher scores)
- Getting hit, letting enemies slip past, or losing a life effectively resets the running score for that life; the agent maximizes total score before all lives are lost.

### The Algorithm: DQN

- 3-layer convolutional network → fully connected → 18 action Q-values  
- Target network updated every 10,000 steps  
- Replay buffer of size 300,000  
- Epsilon-greedy exploration schedule  
- Manual TD target computation  
- 4-frame stacking for motion awareness  

## How to Use

### Train

```
python main.py
```

### Resume Training

```
python main.py --resume
```

### Test Agent

```
python main.py --test
```

Disable rendering:

```
python main.py --test --no-render
```

Test for specific number of episodes:

```
python main.py --test --episodes 20
```

## What Gets Saved

After training, you'll see these files:

- **`model.pth`** - The trained agent (best model)
- **`train_plot.png`** - Graph showing learning progress 
- **`checkpoints/`** – Saved progress (for resuming training)
-  various metadata and pkl files which store progress for checkpoints/plotting


## References
 
- **Environment:** [ALE Krull Documentation](https://ale.farama.org/environments/krull/)
- **Course:** CSCI6353 Reinforcement Learning

---

**Author:** Jared Soto  
**Assignment:** CSCI6353 Homework 3, Part 2  
**Date:** November 2025
