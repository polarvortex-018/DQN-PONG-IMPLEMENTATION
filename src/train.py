"""
Training Script for DQN Pong Agent

This is the main training loop that brings everything together:
- Environment
- Replay Buffer
- DQN Agent
- Training logic

Run this file to start training your agent!
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import json

# Import our modules
import config
from environment import PongEnvironment
from replay_buffer import ReplayBuffer
from dqn_model import DQNAgent


class Trainer:
    """
    Trainer class that manages the entire training process.
    """
    
    def __init__(self, resume_from=None):
        """
        Initialize the trainer.
        
        Args:
            resume_from: Path to checkpoint to resume from (optional)
        """
        print("=" * 80)
        print("DQN PONG AGENT TRAINER")
        print("=" * 80)
        
        # Set up TensorFlow
        self._setup_tensorflow()
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        print("\nInitializing components...")
        self.env = PongEnvironment()
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        self.agent = DQNAgent(
            self.env.get_state_shape(),
            self.env.get_num_actions()
        )
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.avg_q_values = []
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print("\nTrainer initialized successfully!")
        config.print_config()
    
    def _setup_tensorflow(self):
        """Configure TensorFlow settings."""
        # Enable GPU memory growth
        if config.GPU_MEMORY_GROWTH:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU detected: {len(gpus)} device(s)")
                except RuntimeError as e:
                    print(f"GPU setup error: {e}")
            else:
                print("No GPU detected, using CPU")
        
        # Enable mixed precision for faster training on RTX 3050
        if config.USE_MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision training enabled")
    
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        if config.USE_TENSORBOARD:
            os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
    
    def train(self):
        """
        Main training loop.
        
        This is where the magic happens!
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Training for {config.MAX_EPISODES} episodes")
        print(f"Press Ctrl+C to stop training and save progress\n")
        
        try:
            # Training loop
            for episode in range(self.episode, config.MAX_EPISODES):
                self.episode = episode
                
                # Run one episode
                episode_reward, episode_length, avg_loss = self._run_episode()
                
                # Store metrics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                if avg_loss is not None:
                    self.losses.append(avg_loss)
                
                # Print progress
                if episode % config.PRINT_FREQ == 0:
                    self._print_progress()
                
                # Save checkpoint
                if episode % config.SAVE_FREQ == 0 and episode > 0:
                    self.save_checkpoint(f"episode_{episode}")
                
                # Evaluate agent
                if episode % config.EVAL_FREQ == 0 and episode > 0:
                    self._evaluate()
                
                # Save logs
                if episode % config.LOG_FREQ == 0:
                    self._save_logs()
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            self.save_checkpoint("interrupted")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        self._print_final_stats()
    
    def _run_episode(self):
        """
        Run one episode of training.
        
        Returns:
            episode_reward: Total reward for episode
            episode_length: Number of steps in episode
            avg_loss: Average loss during episode
        """
        # Reset environment
        state = self.env.reset()
        done = False
        episode_losses = []
        
        # Get epsilon for this episode
        epsilon = config.get_epsilon(self.episode)
        
        # Episode loop
        while not done and self.env.episode_length < config.MAX_STEPS_PER_EPISODE:
            # Select action
            action = self.agent.get_action(state, epsilon)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience (uint8 single frame for memory efficiency)
            frame_uint8 = self.env.get_last_frame_uint8()
            self.replay_buffer.add(frame_uint8, action, reward, done)
            
            # Train if we have enough experiences
            if self.replay_buffer.is_ready(config.MIN_REPLAY_SIZE):
                loss = self._train_step()
                episode_losses.append(loss)
            
            # Update target network periodically
            if self.total_steps % config.TARGET_UPDATE_FREQ == 0:
                self.agent.update_target_network()
                if config.VERBOSE:
                    print(f"  [Step {self.total_steps}] Target network updated")
            
            # Move to next state
            state = next_state
            self.total_steps += 1
        
        # Calculate average loss
        avg_loss = np.mean(episode_losses) if episode_losses else None
        
        return self.env.episode_reward, self.env.episode_length, avg_loss
    
    def _train_step(self):
        """
        Perform one training step.
        
        Returns:
            Loss value
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            config.BATCH_SIZE
        )
        
        # Train agent
        loss = self.agent.train_step(states, actions, rewards, next_states, dones)
        
        return loss.numpy()
    
    def _evaluate(self):
        """
        Evaluate the agent without exploration.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION (Episode {self.episode})")
        print(f"{'='*60}")
        
        eval_rewards = []
        
        for i in range(config.EVAL_EPISODES):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Greedy action (no exploration)
                action = self.agent.get_action(state, epsilon=0.0)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            print(f"  Eval game {i+1}: {episode_reward:.0f}")
        
        avg_eval_reward = np.mean(eval_rewards)
        print(f"\nAverage evaluation reward: {avg_eval_reward:.2f}")
        
        # Save best model
        if avg_eval_reward > self.best_reward:
            self.best_reward = avg_eval_reward
            self.agent.save(config.BEST_MODEL_PATH)
            print(f"New best model saved! (reward: {avg_eval_reward:.2f})")
        
        print(f"{'='*60}\n")
    
    def _print_progress(self):
        """Print training progress."""
        recent_rewards = self.episode_rewards[-config.PRINT_FREQ:]
        recent_lengths = self.episode_lengths[-config.PRINT_FREQ:]
        recent_losses = self.losses[-config.PRINT_FREQ:] if self.losses else [0]
        
        epsilon = config.get_epsilon(self.episode)
        
        print(f"\nEpisode {self.episode}/{config.MAX_EPISODES}")
        print(f"  Avg Reward: {np.mean(recent_rewards):7.2f} | "
              f"Avg Length: {np.mean(recent_lengths):6.1f} | "
              f"Avg Loss: {np.mean(recent_losses):6.4f}")
        print(f"  Epsilon: {epsilon:.4f} | "
              f"Buffer: {len(self.replay_buffer):6d} | "
              f"Steps: {self.total_steps:7d}")
    
    def _print_final_stats(self):
        """Print final training statistics."""
        print(f"\nFinal Statistics:")
        print(f"  Total episodes: {self.episode}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"  Final avg reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"  Buffer size: {len(self.replay_buffer)}")
    
    def _save_logs(self):
        """Save training logs to file."""
        log_path = os.path.join(config.LOG_DIR, 'training_log.json')
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_native_types(obj):
            """Convert numpy types to native Python types."""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            else:
                return obj
        
        logs = {
            'episode_rewards': convert_to_native_types(self.episode_rewards),
            'episode_lengths': convert_to_native_types(self.episode_lengths),
            'losses': convert_to_native_types(self.losses),
            'total_steps': int(self.total_steps),
            'episodes_completed': int(self.episode),
            'best_reward': float(self.best_reward),
        }
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def save_checkpoint(self, name):
        """
        Save training checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"dqn_pong_{name}.weights.h5")
        self.agent.save(checkpoint_path)
        
        # Save training state
        state_path = os.path.join(config.CHECKPOINT_DIR, f"training_state_{name}.json")
        state = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        self.agent.load(path)
        
        # Try to load training state
        # Handle both .h5 and .weights.h5 extensions
        state_path = path.replace('.weights.h5', '_state.json').replace('.h5', '_state.json')
        
        # Also try the training_state_episode_N.json format
        import re
        match = re.search(r'episode_(\d+)', path)
        if match:
            episode_num = match.group(1)
            alt_state_path = os.path.join(os.path.dirname(path), f'training_state_episode_{episode_num}.json')
            if os.path.exists(alt_state_path):
                state_path = alt_state_path
        
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.episode = state['episode']
            self.total_steps = state['total_steps']
            self.best_reward = state['best_reward']
            print(f"Resumed from episode {self.episode}")
        else:
            print(f"Warning: Could not find training state file at {state_path}")
            print("Starting episode counter from 0")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent on Pong')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to train (overrides config)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.episodes:
        config.MAX_EPISODES = args.episodes
    
    # Create trainer
    trainer = Trainer(resume_from=args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
