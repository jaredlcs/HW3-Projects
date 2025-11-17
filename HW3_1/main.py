"""
HW3-1: RL on Gymnasium (LunarLander-v3 Discrete)
CSCI6353 Homework 3
Author: Jared Soto
Date: November 2025

This script implements PPO (Proximal Policy Optimization) for LunarLander-v3.
Includes checkpoint functionality for training on Google Colab.

Target: Average reward >= 200 (stable performance)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for LunarLander PPO experiment"""
    # Environment
    ENV_NAME = "LunarLander-v3"
    TARGET_REWARD = 200  # Target average reward
    
    # Training
    MAX_EPISODES = 5000  # Maximum episodes for training
    T_HORIZON = 4096  # Steps per policy update (PPO batch size)
    K_EPOCHS = 10  # Number of epochs for PPO update
    BATCH_SIZE = 64  # Mini-batch size for PPO
    
    # Network Architecture
    STATE_DIM = 8  # LunarLander observation space
    HIDDEN_DIM = 128  # Hidden layer size
    ACTION_DIM = 4  # LunarLander discrete actions (do nothing, left, main, right)
    
    # PPO Hyperparameters
    LEARNING_RATE = 0.0003  # Learning rate (will decay)
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE parameter
    CLIP_EPSILON = 0.2  # PPO clipping parameter
    ENTROPY_COEF = 0.01  # Entropy bonus coefficient (lower for stability)
    VALUE_COEF = 0.5  # Value loss coefficient
    MAX_GRAD_NORM = 0.5  # Gradient clipping
    VALUE_CLIP = 0.1  # Value function clipping (reduced)
    
    # Checkpoint settings
    CHECKPOINT_DIR = Path("checkpoints")
    MODEL_PATH = Path("model.pth")
    PLOT_PATH = Path("train_plot.png")
    BEST_RUN_PATH = Path("best_run.pkl")
    ALL_RUNS_PATH = Path("all_runs.pkl")
    
    # Performance tracking
    EVAL_WINDOW = 100  # Episodes to average for convergence check
    
    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*60}")
        print("HW3-1: LunarLander-v3 with PPO")
        print(f"{'='*60}")
        print(f"Environment: {self.ENV_NAME}")
        print(f"Target Average Reward: {self.TARGET_REWARD}")
        print(f"Max Episodes: {self.MAX_EPISODES}")
        print(f"Device: {self.device}")
        print(f"\nPPO Hyperparameters:")
        print(f"  - Learning Rate: {self.LEARNING_RATE}")
        print(f"  - Gamma: {self.GAMMA}")
        print(f"  - GAE Lambda: {self.GAE_LAMBDA}")
        print(f"  - Clip Epsilon: {self.CLIP_EPSILON}")
        print(f"  - T Horizon: {self.T_HORIZON}")
        print(f"  - K Epochs: {self.K_EPOCHS}")
        print(f"  - Hidden Dim: {self.HIDDEN_DIM}")
        print(f"{'='*60}\n")


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages saving and loading training progress"""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_file = config.CHECKPOINT_DIR / "training_progress.pkl"
        self.metadata_file = config.CHECKPOINT_DIR / "metadata.json"
    
    def save_checkpoint(self, episode, model, optimizer, scores, scheduler=None, metadata=None):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scores': scores,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        if metadata:
            # Convert numpy types to Python native types for JSON
            metadata_json = {k: (bool(v) if isinstance(v, (np.bool_, bool)) else 
                                float(v) if isinstance(v, (np.floating, float)) else 
                                int(v) if isinstance(v, (np.integer, int)) else v) 
                           for k, v in metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_json, f, indent=2)
        
        print(f"✓ Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self):
        """Load training checkpoint if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"✓ Loaded checkpoint from episode {checkpoint['episode']}")
            return checkpoint
        return None
    
    def clear_checkpoints(self):
        """Clear all saved checkpoints"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("✓ All checkpoints cleared")
    
    def save_best_run(self, scores, avg_score, run_number):
        """Save this run if it's the best so far"""
        best_run_file = self.config.BEST_RUN_PATH
        
        # Load previous best if exists
        if best_run_file.exists():
            with open(best_run_file, 'rb') as f:
                best_data = pickle.load(f)
                best_avg = best_data['avg_score']
        else:
            best_avg = -float('inf')
        
        # Save if this run is better
        if avg_score > best_avg:
            best_data = {
                'scores': scores,
                'avg_score': avg_score,
                'run_number': run_number,
                'timestamp': datetime.now().isoformat()
            }
            with open(best_run_file, 'wb') as f:
                pickle.dump(best_data, f)
            print(f"\n{'='*60}")
            print(f"🏆 NEW BEST RUN! Run #{run_number} with avg score: {avg_score:.2f}")
            print(f"{'='*60}\n")
            return True
        return False
    
    def load_best_run(self):
        """Load the best run data"""
        best_run_file = self.config.BEST_RUN_PATH
        if best_run_file.exists():
            with open(best_run_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_run_history(self, scores, avg_score, run_number):
        """Append this run to history"""
        history_file = self.config.ALL_RUNS_PATH
        
        # Load existing history
        if history_file.exists():
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
        else:
            history = []
        
        # Append current run
        history.append({
            'run_number': run_number,
            'avg_score': avg_score,
            'episodes': len(scores),
            'timestamp': datetime.now().isoformat()
        })
        
        # Save updated history
        with open(history_file, 'wb') as f:
            pickle.dump(history, f)
    
    def get_next_run_number(self):
        """Get the next run number"""
        history_file = self.config.ALL_RUNS_PATH
        if history_file.exists():
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
                return len(history) + 1
        return 1
    
    def clear_all(self):
        """Clear all saved data including best run"""
        self.clear_checkpoints()
        if self.config.BEST_RUN_PATH.exists():
            self.config.BEST_RUN_PATH.unlink()
        if self.config.ALL_RUNS_PATH.exists():
            self.config.ALL_RUNS_PATH.unlink()
        print("✓ All run data cleared")


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic Network for A2C"""
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """Forward pass through network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy distribution with numerical stability
        logits = self.actor(x)
        # Clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-20, max=20)
        action_probs = F.softmax(logits, dim=-1)
        # Clamp probabilities to prevent NaN
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        
        # State value
        value = self.critic(x)
        
        return action_probs, value


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOMemory:
    """Memory buffer for PPO training"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """Clear all stored data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_batch(self):
        """Get all stored data as tensors"""
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones
        )


class PPOAgent:
    """PPO (Proximal Policy Optimization) Agent"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize network
        self.model = ActorCritic(
            config.STATE_DIM,
            config.HIDDEN_DIM,
            config.ACTION_DIM
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Learning rate scheduler (step decay for stability)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,  # Decay every 500 episodes
            gamma=0.95  # Multiply LR by 0.95
        )
        
        # Memory buffer
        self.memory = PPOMemory()
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.model(state)
        
        # Sample action from distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        # Add next value for bootstrapping
        next_value = 0
        values = values + [next_value]
        
        # Compute GAE in reverse
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.config.GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO algorithm"""
        # Get batch from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get_batch()
        
        # Store reference to agent for value clipping
        agent = self
        
        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages with stability
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8 and not torch.isnan(adv_std):
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean
        
        # PPO update for K epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.K_EPOCHS):
            # Generate random mini-batches
            batch_size = len(states)
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.config.BATCH_SIZE):
                end = start + self.config.BATCH_SIZE
                batch_idx = indices[start:end]
                
                # Get mini-batch
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Forward pass
                action_probs, values = self.model(batch_states)
                dist = Categorical(action_probs)
                
                # Get log probs for taken actions
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.CLIP_EPSILON, 1.0 + self.config.CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss - ensure same shape
                values = values.squeeze(-1)
                batch_returns = batch_returns.squeeze(-1) if batch_returns.dim() > 1 else batch_returns
                
                # Simple MSE loss (more stable than clipped version)
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Check for NaN
                if torch.isnan(actor_loss) or torch.isnan(critic_loss) or torch.isnan(entropy):
                    print(f"Warning: NaN detected in losses. Skipping this batch.")
                    continue
                
                # Total loss
                loss = actor_loss + self.config.VALUE_COEF * critic_loss - self.config.ENTROPY_COEF * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory
        self.memory.clear()
        
        num_updates = self.config.K_EPOCHS * (len(states) // self.config.BATCH_SIZE + 1)
        return total_actor_loss / num_updates, total_critic_loss / num_updates, total_entropy / num_updates
    
    def save_model(self, path):
        """Save model to file"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path):
        """Load model from file"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✓ Model loaded from {path}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(config, resume=False):
    """Train PPO agent on LunarLander"""
    
    # Initialize environment
    env = gym.make(config.ENV_NAME)
    
    # Initialize agent
    agent = PPOAgent(config)
    
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(config)
    
    # Get run number
    run_number = checkpoint_mgr.get_next_run_number() if not resume else checkpoint_mgr.get_next_run_number() - 1
    
    # Load checkpoint if resuming
    start_episode = 0
    scores = []
    total_steps = 0
    
    if resume:
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            start_episode = checkpoint['episode']
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scores = checkpoint['scores']
            total_steps = checkpoint.get('total_steps', 0)
            print(f"Resuming from episode {start_episode}")
    
    # Training loop
    print(f"\nStarting training from episode {start_episode}...")
    best_avg_score = -float('inf')
    
    episode = start_episode
    while episode < config.MAX_EPISODES:
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Collect trajectory
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.memory.store(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            total_steps += 1
            state = next_state
            
            if done:
                break
        
        # Record score
        scores.append(episode_reward)
        episode += 1
        
        # Update policy after collecting T_HORIZON steps
        if len(agent.memory.states) >= config.T_HORIZON:
            actor_loss, critic_loss, entropy = agent.update()
            # Step learning rate scheduler
            agent.scheduler.step()
            # Memory is cleared inside update()
        
        # Calculate moving average
        if len(scores) >= config.EVAL_WINDOW:
            avg_score = np.mean(scores[-config.EVAL_WINDOW:])
        else:
            avg_score = np.mean(scores)
        
        # Print progress
        if episode % 10 == 0:
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"Episode {episode}/{config.MAX_EPISODES} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Avg Score ({config.EVAL_WINDOW}): {avg_score:.2f} | "
                  f"Steps: {total_steps} | "
                  f"LR: {current_lr:.6f}")
        
        # Save checkpoint every 50 episodes
        if episode % 50 == 0:
            checkpoint_mgr.save_checkpoint(
                episode,
                agent.model,
                agent.optimizer,
                scores,
                scheduler=agent.scheduler,
                metadata={
                    'episode': episode,
                    'total_steps': total_steps,
                    'avg_score': float(avg_score),
                    'best_avg_score': float(best_avg_score)
                }
            )
            
        # Save model only if significantly better (reduce spam)
        if avg_score > best_avg_score + 1.0:  # Require 1.0 improvement
            best_avg_score = avg_score
            agent.save_model(config.MODEL_PATH)
        
        # Check convergence
        if len(scores) >= config.EVAL_WINDOW and avg_score >= config.TARGET_REWARD:
            print(f"\n{'='*60}")
            print(f"✓ Target reached! Average reward: {avg_score:.2f}")
            print(f"{'='*60}\n")
            agent.save_model(config.MODEL_PATH)  # Save final converged model
            break
    
    env.close()
    
    # Calculate final average score
    final_avg_score = np.mean(scores[-config.EVAL_WINDOW:]) if len(scores) >= config.EVAL_WINDOW else np.mean(scores)
    
    # Save run history
    checkpoint_mgr.save_run_history(scores, final_avg_score, run_number)
    
    # Check if this is the best run
    is_best = checkpoint_mgr.save_best_run(scores, final_avg_score, run_number)
    
    # Save final checkpoint
    checkpoint_mgr.save_checkpoint(
        episode,
        agent.model,
        agent.optimizer,
        scores,
        scheduler=agent.scheduler,
        metadata={
            'episode': episode,
            'total_steps': total_steps,
            'final_avg_score': float(final_avg_score),
            'best_avg_score': float(best_avg_score),
            'converged': bool(final_avg_score >= config.TARGET_REWARD),
            'run_number': run_number,
            'is_best_run': is_best
        }
    )
    
    # Always plot the BEST run (not necessarily this one)
    best_run_data = checkpoint_mgr.load_best_run()
    if best_run_data:
        print(f"\nPlotting best run (Run #{best_run_data['run_number']}) with avg score: {best_run_data['avg_score']:.2f}")
        plot_training_curve(best_run_data['scores'], config, run_number=best_run_data['run_number'])
    else:
        plot_training_curve(scores, config, run_number=run_number)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Run #{run_number} Summary:")
    print(f"  Final Avg Score: {final_avg_score:.2f}")
    print(f"  Episodes: {episode}")
    print(f"  Converged: {'Yes' if final_avg_score >= config.TARGET_REWARD else 'No'}")
    if best_run_data:
        print(f"\nBest Run Overall: Run #{best_run_data['run_number']} (Avg: {best_run_data['avg_score']:.2f})")
    print(f"{'='*60}\n")
    
    return scores


def plot_training_curve(scores, config, run_number=None):
    """Plot and save training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.6, label='Episode Reward')
    
    # Plot moving average
    if len(scores) >= config.EVAL_WINDOW:
        moving_avg = [np.mean(scores[max(0, i-config.EVAL_WINDOW):i+1]) 
                      for i in range(len(scores))]
        plt.plot(moving_avg, linewidth=2, label=f'{config.EVAL_WINDOW}-Episode Moving Avg')
    
    plt.axhline(y=config.TARGET_REWARD, color='r', linestyle='--', 
                label=f'Target Reward ({config.TARGET_REWARD})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Add run number to title if provided
    title = 'LunarLander-v3 Training Curve (PPO)'
    if run_number:
        title += f' - Best Run #{run_number}'
    plt.title(title)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=300)
    print(f"✓ Training plot saved to {config.PLOT_PATH}")
    plt.close()


# ============================================================================
# TEST MODE
# ============================================================================

def test(config, num_episodes=10, render=True):
    """Test trained model"""
    
    # Check if model exists
    if not config.MODEL_PATH.exists():
        print(f"Error: Model file not found at {config.MODEL_PATH}")
        print("Please train the model first.")
        return
    
    # Initialize environment with rendering
    if render:
        env = gym.make(config.ENV_NAME, render_mode="human")
    else:
        env = gym.make(config.ENV_NAME)
    
    # Initialize and load agent
    agent = PPOAgent(config)
    agent.load_model(config.MODEL_PATH)
    
    print(f"\nTesting model for {num_episodes} episodes...")
    test_scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Select action (greedy)
            action, _, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        test_scores.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"Test Results ({num_episodes} episodes):")
    print(f"  Mean Reward: {np.mean(test_scores):.2f}")
    print(f"  Std Reward: {np.std(test_scores):.2f}")
    print(f"  Min Reward: {np.min(test_scores):.2f}")
    print(f"  Max Reward: {np.max(test_scores):.2f}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HW3-1: LunarLander-v3 with PPO')
    parser.add_argument('--test', action='store_true', 
                        help='Test trained model instead of training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of episodes for testing (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering during test')
    parser.add_argument('--clear-all', action='store_true',
                        help='Clear all saved data including best run')
    parser.add_argument('--show-runs', action='store_true',
                        help='Show history of all training runs')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    checkpoint_mgr = CheckpointManager(config)
    
    if args.show_runs:
        # Show run history
        if config.ALL_RUNS_PATH.exists():
            with open(config.ALL_RUNS_PATH, 'rb') as f:
                history = pickle.load(f)
            print(f"\n{'='*60}")
            print("Training Run History:")
            print(f"{'='*60}")
            for run in history:
                print(f"Run #{run['run_number']}: Avg Score = {run['avg_score']:.2f}, Episodes = {run['episodes']}")
            best_run = checkpoint_mgr.load_best_run()
            if best_run:
                print(f"\n🏆 Best: Run #{best_run['run_number']} with avg score {best_run['avg_score']:.2f}")
            print(f"{'='*60}\n")
        else:
            print("No training history found.")
        return
    
    if args.clear_all:
        checkpoint_mgr.clear_all()
        return
    
    if args.test:
        # Test mode
        test(config, num_episodes=args.test_episodes, render=not args.no_render)
    else:
        # Training mode
        if args.resume:
            train(config, resume=True)
        else:
            # Ask to clear checkpoints
            response = input("Clear previous checkpoints? (y/N): ").strip().lower()
            if response == 'y':
                checkpoint_mgr.clear_checkpoints()
            
            train(config, resume=False)


if __name__ == "__main__":
    main()
