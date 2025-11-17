# HW3-1: LunarLander-v3 with PPO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaredlcs/HW3-Projects/blob/main/HW3_1/colab_notebook.ipynb)

**CSCI6353 Homework 3 - Part 1**  
**Author:** Jared Soto  
**Date:** November 2025

## Quick Start

### Run in Google Colab (Recommended)

**Click the "Open in Colab" badge above** to open a ready-to-run notebook, or manually use:

```python
# Install Box2D dependencies
!apt-get update -qq
!apt-get install -y swig build-essential python3-dev

# Clone and setup
!git clone https://github.com/jaredlcs/HW3-Projects.git
%cd HW3-Projects/HW3_1
!pip install gymnasium[box2d] torch matplotlib

# Train
!python main.py
```

See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed Colab instructions.

### Run Locally

```bash
git clone https://github.com/jaredlcs/HW3-Projects.git
cd HW3-Projects/HW3_1
pip install -r requirements.txt
python main.py
```

## Overview

This project implements **PPO (Proximal Policy Optimization)** to train an agent on the LunarLander-v3 discrete environment from Gymnasium. The agent learns to safely land a lunar module by controlling its thrusters.

**Target Performance:** Average reward ≥ 200 (stable landing)

## Environment Details

- **Environment:** `LunarLander-v3` (Discrete)
- **Observation Space:** 8-dimensional continuous (position, velocity, angle, angular velocity, leg contact)
- **Action Space:** 4 discrete actions
  - 0: Do nothing
  - 1: Fire left orientation engine
  - 2: Fire main engine
  - 3: Fire right orientation engine
- **Reward:** Ranges from approximately -100 to +200
  - Crash: -100
  - Successful landing: +100-200
  - Fuel consumption penalized

## Algorithm: PPO (Proximal Policy Optimization)

PPO is a state-of-the-art policy gradient method that:
- **Actor:** Learns a policy π(a|s) that maps states to action probabilities
- **Critic:** Learns a value function V(s) to estimate expected returns
- **Clipped Objective:** Prevents large policy updates using ratio clipping
- **GAE:** Uses Generalized Advantage Estimation for variance reduction

### Key Features
- Clipped surrogate objective for stable training
- GAE (λ=0.95) for better advantage estimation
- Multiple epochs per batch for sample efficiency
- Entropy bonus for exploration
- Gradient clipping for training stability
- Shared network backbone with separate actor/critic heads

## Network Architecture

```
Input (8) → FC(128) → ReLU → FC(128) → ReLU
                                        ↓
                         ┌──────────────┴──────────────┐
                         ↓                             ↓
                   Actor Head (4)              Critic Head (1)
                  [Action Probs]                  [Value]
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0003 | Adam optimizer learning rate |
| Gamma (γ) | 0.99 | Discount factor |
| GAE Lambda (λ) | 0.95 | GAE advantage estimation parameter |
| Clip Epsilon (ε) | 0.2 | PPO clipping parameter |
| T Horizon | 2048 | Steps per policy update |
| K Epochs | 4 | PPO update epochs per batch |
| Batch Size | 64 | Mini-batch size for PPO |
| Entropy Coef | 0.01 | Entropy bonus coefficient |
| Value Coef | 0.5 | Value loss coefficient |
| Hidden Dim | 128 | Hidden layer size |
| Max Episodes | 2000 | Maximum training episodes |
| Target Reward | 200 | Convergence threshold (100-ep avg) |

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

### Quick Setup

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch gymnasium numpy matplotlib
```

### Google Colab Setup

```python
# Install Box2D system dependencies
!apt-get update -qq
!apt-get install -y swig build-essential python3-dev

# Clone the repository
!git clone https://github.com/jaredlcs/HW3-Projects.git
%cd HW3-Projects/HW3_1

# Install Python dependencies
!pip install gymnasium[box2d] torch matplotlib

# Run training
!python main.py
```

## Usage

### Training

Train a new model from scratch:

```bash
python main.py
```

Resume training from checkpoint:

```bash
python main.py --resume
```

### Testing

Test the trained model (requires `model.pth`):

```bash
python main.py --test
```

Test without rendering (faster):

```bash
python main.py --test --no-render
```

Test for 20 episodes:

```bash
python main.py --test --test-episodes 20
```

## Checkpoint System

The checkpoint system allows you to:
- **Resume training** if interrupted (e.g., Colab timeout)
- **Save progress** every 50 episodes
- **Track best model** based on 100-episode moving average

### Checkpoint Files

- `checkpoints/training_progress.pkl` - Full training state (model, optimizer, scores)
- `checkpoints/metadata.json` - Training metadata and statistics
- `model.pth` - Best model (highest 100-episode average)
- `train_plot.png` - Training curve visualization

### Resuming After Interruption

If training is interrupted:

1. Simply run: `python main.py --resume`
2. Training will continue from the last checkpoint
3. All episode scores are preserved

## Output Files

After training, you will have:

1. **`model.pth`** - Trained model weights (best performing)
2. **`train_plot.png`** - Learning curve showing training progress
3. **`checkpoints/`** - Training checkpoints for resuming

## Expected Training Time

- **CPU:** ~20-40 minutes for convergence
- **GPU:** ~10-20 minutes for convergence
- **Convergence:** Typically 300-600 episodes to reach avg reward ≥ 200 (faster than A2C!)

## Training Curve

The `train_plot.png` shows:
- Raw episode rewards (light blue)
- 100-episode moving average (dark blue)
- Target reward threshold (red dashed line at 200)

**Expected Trend:** Upward trajectory with convergence around 200-250 average reward

## Demo Video

**YouTube Link:** [Will be added after recording]

The demo video shows:
- Successful lunar landings using the trained model
- Smooth thruster control
- Safe touchdown with minimal fuel consumption

## Code Structure

```
main.py
├── Config                  # Hyperparameters and settings
├── CheckpointManager       # Save/load training progress
├── ActorCritic            # Neural network (actor + critic)
├── A2CAgent               # Agent with training logic
├── train()                # Training loop with checkpointing
├── test()                 # Model evaluation
└── main()                 # Entry point with CLI args
```

## Implementation Notes

### What's Implemented (No External RL Libraries)
- ✅ Actor-Critic network architecture
- ✅ PPO clipped surrogate objective
- ✅ GAE (Generalized Advantage Estimation)
- ✅ Multiple epoch updates with mini-batches
- ✅ Entropy bonus for exploration
- ✅ Advantage calculation and normalization
- ✅ Experience collection and buffer management
- ✅ Gradient clipping for stability

### External Libraries Used (Allowed)
- `PyTorch` - Neural network implementation (allowed per assignment)
- `Gymnasium` - Environment only (not RL algorithm)
- Standard Python libraries (NumPy, Matplotlib, etc.)

**No Stable-Baselines3, RLlib, or high-level RL frameworks used.**

## Troubleshooting

### Model doesn't converge
- Try increasing `MAX_EPISODES` to 3000-5000
- Adjust `LEARNING_RATE` (try 0.0002 or 0.0005)
- Increase `HIDDEN_DIM` to 256
- Adjust `CLIP_EPSILON` (try 0.1 or 0.3)
- Increase `T_HORIZON` to 4096 for more samples per update

### Training is slow
- Use GPU if available (automatically detected)
- Reduce checkpoint frequency (modify save interval)

### Colab session timeout
- Use `--resume` flag when restarting
- Checkpoints are saved every 50 episodes

### Can't render in test mode
- Make sure you have a display (won't work on headless servers)
- Use `--no-render` flag for testing without visualization

## Performance Metrics

Expected final performance:
- **Mean Reward:** 200-250
- **Success Rate:** >85% successful landings
- **Convergence:** 300-600 episodes (PPO is more sample efficient!)

## Assignment Deliverables Checklist

- [x] `main.py` - Training and testing script
- [x] `README.md` - This file with all details
- [ ] `model.pth` - Trained model (generated during training)
- [ ] `train_plot.png` - Learning curve (generated during training)
- [ ] YouTube demo video link (record after training)

## How to Generate All Deliverables

```bash
# 1. Train the model
python main.py

# 2. This creates: model.pth and train_plot.png

# 3. Test the model and record video
python main.py --test

# 4. Upload video to YouTube (public/unlisted)

# 5. Add YouTube link to this README
```

## References

- **Environment:** [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- **Algorithm:** PPO based on "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **GAE:** "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- **Assignment:** CSCI6353 Homework 3, Part 1

## Contact

For questions or issues:
- Check code comments in `main.py`
- Verify all dependencies are installed
- Ensure Python 3.8+ is being used

---

**Note:** This implementation follows the assignment requirement of implementing RL logic from scratch without using high-level RL libraries like Stable-Baselines3.
