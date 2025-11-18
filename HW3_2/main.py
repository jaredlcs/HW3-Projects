"""
HW3-2: DQN on Krull (ALE/Krull-v5)
CSCI6353 Homework 3
Author: Jared Soto
Date: November 2025
This script implements DQN (Deep Q-Network) for ALE/Krull-v5.
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime
import random
from collections import deque

# Register ALE
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not installed.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    ### UPDATED FOR KRULL
    ENV_NAME = "ALE/Krull-v5"
    
    MAX_STEPS = 2_000_000
    LEARNING_STARTS = 50_000
    UPDATE_FREQ = 4
    TARGET_UPDATE_FREQ = 10_000
    
    BUFFER_SIZE = 300_000
    BATCH_SIZE = 32
    
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_STEPS = 1_000_000
    MAX_GRAD_NORM = 10.0
    
    LOG_INTERVAL = 10_000
    SAVE_INTERVAL = 50_000
    
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pkl"
    MODEL_PATH = Path("model.pth")
    MODEL_METADATA_PATH = Path("model_metadata.json")
    PLOT_PATH = Path("train_plot.png")
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        print("\nInitialized DQN Config for KRULL\n")


# ============================================================================
# FRAME PREPROCESSING
# ============================================================================

def preprocess_frame(frame):
    """Convert to grayscale, downsample to 84x84, and normalize to [0, 1]."""
    
    if len(frame.shape) == 3:
        gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    else:
        gray = frame

    h, w = gray.shape
    sh = max(h // 84, 1)
    sw = max(w // 84, 1)

    resized = gray[::sh, ::sw][:84, :84]
    return (resized / 255.0).astype(np.float32)


class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        processed = preprocess_frame(frame)
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)
    
    def push(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN MODEL
# ============================================================================

class DQN(nn.Module):
    def __init__(self, num_frames=4, num_actions=18):
        super().__init__()
        self.conv1 = nn.Conv2d(num_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================================
# AGENT
# ============================================================================

class DQNAgent:
    def __init__(self, config, num_actions):
        self.config = config
        self.device = config.device
        self.num_actions = num_actions
        
        self.q_network = DQN(num_frames=4, num_actions=num_actions).to(self.device)
        self.target_network = DQN(num_frames=4, num_actions=num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY_STEPS
    
    def select_action(self, state, eval_mode=False):
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.q_network(t)
                return q.argmax().item()
        return random.randrange(self.num_actions)
    
    def update_epsilon(self):
        self.epsilon = max(self.config.EPSILON_END, self.epsilon - self.epsilon_decay)
    
    def train_step(self):
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.config.GAMMA * next_q
        
        loss = F.smooth_l1_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.MAX_GRAD_NORM)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(config, resume=False):
    env = gym.make(config.ENV_NAME)
    num_actions = env.action_space.n
    agent = DQNAgent(config, num_actions)
    
    frame_stack = FrameStack()
    
    start_step = 0
    episode_rewards = []
    episode_steps = []
    training_losses = []
    best_avg_reward = -float("inf")
    best_step = 0

    if config.MODEL_METADATA_PATH.exists():
        try:
            best_data = json.loads(config.MODEL_METADATA_PATH.read_text())
            best_avg_reward = float(best_data.get("avg_reward", best_avg_reward))
            best_step = int(best_data.get("step", best_step))
            if np.isfinite(best_avg_reward):
                print(f"Best model so far: Avg100 = {best_avg_reward:.2f} at step {best_step:,}")
        except Exception as exc:
            print(f"Warning: failed to read best model metadata ({exc}).")

    if resume:
        checkpoint = load_checkpoint(config)
        if checkpoint is not None:
            start_step = checkpoint.get("step", 0)
            agent.q_network.load_state_dict(checkpoint["q_network"])
            agent.target_network.load_state_dict(checkpoint["target_network"])
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
            agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
            episode_rewards = list(checkpoint.get("episode_rewards", []))
            episode_steps = list(checkpoint.get("episode_steps", []))
            training_losses = list(checkpoint.get("training_losses", []))
            if episode_rewards and len(episode_steps) != len(episode_rewards):
                approx_max = max(start_step, len(episode_rewards))
                episode_steps = np.linspace(
                    1,
                    max(1, approx_max),
                    num=len(episode_rewards),
                    dtype=np.int64
                ).tolist()
            print(
                f"Resuming training from step {start_step:,} "
                f"({len(episode_rewards)} episodes recorded, epsilon={agent.epsilon:.3f})."
            )
        else:
            print("Resume requested but checkpoint not found. Starting from scratch.")
    
    obs, _ = env.reset()
    state = frame_stack.reset(obs)
    ep_reward = 0
    
    for step in range(start_step + 1, config.MAX_STEPS + 1):
        
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = frame_stack.push(next_obs)

        clipped_reward = float(np.clip(reward, -1.0, 1.0))
        agent.replay_buffer.push(state, action, clipped_reward, next_state, done)
        
        state = next_state
        ep_reward += reward
        
        agent.update_epsilon()
        
        if len(agent.replay_buffer) >= config.LEARNING_STARTS and step % config.UPDATE_FREQ == 0:
            loss = agent.train_step()
            if loss is not None:
                training_losses.append(loss)
        
        if step % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        if done:
            episode_rewards.append(ep_reward)
            episode_steps.append(step)
            ep_reward = 0
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
            
            if episode_rewards:
                window = min(len(episode_rewards), 100)
                current_avg = float(np.mean(episode_rewards[-window:]))
                if current_avg > best_avg_reward:
                    best_avg_reward = current_avg
                    best_step = step
                    save_best_model(
                        config,
                        agent,
                        step=step,
                        avg_reward=current_avg,
                        window=window,
                        episodes=len(episode_rewards)
                    )
        
        if step % config.LOG_INTERVAL == 0:
            if episode_rewards:
                reward_window = min(len(episode_rewards), 100)
                avg_reward = float(np.mean(episode_rewards[-reward_window:]))
            else:
                reward_window = 0
                avg_reward = 0.0
            if training_losses:
                loss_window = min(len(training_losses), 100)
                avg_loss = float(np.mean(training_losses[-loss_window:]))
                loss_str = f"Loss{loss_window}: {avg_loss:.4f}"
            else:
                loss_window = 0
                loss_str = "Loss: n/a"
            avg_label = f"Avg{reward_window}" if reward_window else "Avg"
            print(
                f"[Step {step:,}/{config.MAX_STEPS:,}] "
                f"Episodes: {len(episode_rewards)} | "
                f"{avg_label}: {avg_reward:.2f} | "
                f"{loss_str} | "
                f"Eps: {agent.epsilon:.3f}"
            )
        
        if step % config.SAVE_INTERVAL == 0:
            save_checkpoint(config, agent, step, episode_rewards, episode_steps, training_losses)
    
    env.close()
    save_checkpoint(config, agent, config.MAX_STEPS, episode_rewards, episode_steps, training_losses)
    plot_results(config, episode_rewards, episode_steps)


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(config, agent, step, episode_rewards, episode_steps, training_losses):
    checkpoint = {
        "step": step,
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "training_losses": training_losses,
    }
    torch.save(checkpoint, config.CHECKPOINT_PATH)
    print(f"Checkpoint saved at step {step:,}")


def load_checkpoint(config):
    if not config.CHECKPOINT_PATH.exists():
        print("No checkpoint found.")
        return None
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.device)
    step = checkpoint.get("step", 0)
    print(f"Loaded checkpoint from step {step:,}")
    return checkpoint


def clear_checkpoint(config):
    if config.CHECKPOINT_PATH.exists():
        config.CHECKPOINT_PATH.unlink()
        print("Existing checkpoint cleared.")


def save_best_model(config, agent, step, avg_reward, window, episodes):
    torch.save({
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict()
    }, config.MODEL_PATH)
    metadata = {
        "step": step,
        "avg_reward": avg_reward,
        "window": window,
        "episodes": episodes,
        "timestamp": datetime.now().isoformat()
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(
        f"New best model saved to {config.MODEL_PATH} | "
        f"Avg{window}: {avg_reward:.2f} at step {step:,}"
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(config, episode_rewards, episode_steps=None):
    if not episode_rewards:
        print("No episodes to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    if episode_steps is not None and len(episode_steps) == len(episode_rewards):
        x_values = np.array(episode_steps) / 1e6
        plt.xlabel("Environment Step (1e6)")
    else:
        x_values = list(range(1, len(episode_rewards) + 1))
        plt.xlabel("Episode")
    plt.plot(x_values, episode_rewards, alpha=0.3, label="Episode Reward")
    
    if len(episode_rewards) >= 100:
        avg = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
        plt.plot(x_values, avg, linewidth=2, label="100-Episode Moving Average")
    
    plt.ylabel("Reward")
    
    ### UPDATED FOR KRULL
    plt.title("DQN Training on Krull")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=300)
    print(f"Saved plot to {config.PLOT_PATH}")


# ============================================================================
# TESTING
# ============================================================================

def test(config, num_episodes=10, render=True):
    if not config.MODEL_PATH.exists():
        print("Model not found. Train first.")
        return
    
    if render:
        env = gym.make(config.ENV_NAME, render_mode="human")
    else:
        env = gym.make(config.ENV_NAME)
    
    num_actions = env.action_space.n
    agent = DQNAgent(config, num_actions)
    
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.device)
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.q_network.eval()
    
    frame_stack = FrameStack()
    
    print(f"\nTesting for {num_episodes} episodes...\n")
    rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        ep_reward = 0
        
        while True:
            action = agent.select_action(state, eval_mode=True)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            state = frame_stack.push(next_obs)
            ep_reward += reward
            if done:
                break
        
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.1f}")
    
    env.close()
    print("\nAverage:", np.mean(rewards))


# ============================================================================
# MAIN
# ============================================================================

def main():
    ### UPDATED DESCRIPTION
    parser = argparse.ArgumentParser(description="HW3-2: DQN on Krull")
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    
    if args.test:
        test(config, args.episodes, render=not args.no_render)
    else:
        if args.resume:
            train(config, resume=True)
        else:
            checkpoint_exists = config.CHECKPOINT_PATH.exists()
            if checkpoint_exists:
                response = input("Previous checkpoint found. Resume training? (Y/n): ").strip().lower()
                if response in ("n", "no"):
                    clear_checkpoint(config)
                    train(config, resume=False)
                else:
                    train(config, resume=True)
            else:
                train(config, resume=False)


if __name__ == "__main__":
    main()
