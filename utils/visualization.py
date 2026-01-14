"""
Visualization Utilities

This module provides functions to visualize training progress:
- Plot reward curves
- Plot loss curves
- Plot episode lengths
- Create training summary dashboard
"""

import os
import sys

# Add parent directory to path to import config from src
# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')

# Add both to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import config


def load_training_logs(log_dir=None):
    """
    Load training logs from file.
    
    Args:
        log_dir: Directory containing logs (uses config if None)
        
    Returns:
        Dictionary with training metrics
    """
    log_dir = log_dir or config.LOG_DIR
    log_file = os.path.join(log_dir, 'training_log.json')
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    return logs


def smooth_curve(values, window=10):
    """
    Smooth a curve using moving average.
    
    Args:
        values: List of values
        window: Window size for moving average
        
    Returns:
        Smoothed values
    """
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i+1]))
    
    return smoothed


def plot_training_progress(log_dir=None, save_path=None):
    """
    Create a comprehensive training progress dashboard.
    
    Args:
        log_dir: Directory containing logs
        save_path: Path to save plot (shows if None)
    """
    # Load logs
    logs = load_training_logs(log_dir)
    if logs is None:
        return
    
    # Extract data
    episodes = list(range(len(logs['episode_rewards'])))
    rewards = logs['episode_rewards']
    lengths = logs['episode_lengths']
    losses = logs.get('losses', [])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Episode Rewards
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smooth_curve(rewards, 50), color='blue', linewidth=2, label='Smoothed (50 ep)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(episodes, lengths, alpha=0.3, color='green')
    ax2.plot(episodes, smooth_curve(lengths, 50), color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    ax3 = fig.add_subplot(gs[1, 1])
    if losses:
        loss_episodes = list(range(len(losses)))
        ax3.plot(loss_episodes, losses, alpha=0.3, color='red')
        ax3.plot(loss_episodes, smooth_curve(losses, 50), color='red', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No loss data available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Reward Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(rewards, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Statistics Table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics
    recent_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    stats_text = f"""
    TRAINING STATISTICS
    {'='*40}
    
    Total Episodes: {len(rewards)}
    Total Steps: {logs.get('total_steps', 'N/A')}
    
    Rewards:
      Mean (all): {np.mean(rewards):.2f}
      Mean (last 100): {np.mean(recent_100):.2f}
      Best: {np.max(rewards):.0f}
      Worst: {np.min(rewards):.0f}
    
    Episode Length:
      Mean: {np.mean(lengths):.1f}
      Max: {np.max(lengths):.0f}
    
    Best Model Reward: {logs.get('best_reward', 'N/A')}
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('DQN Pong Training Progress', fontsize=16, fontweight='bold')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rewards_only(log_dir=None, save_path=None):
    """
    Create a simple reward plot.
    
    Args:
        log_dir: Directory containing logs
        save_path: Path to save plot (shows if None)
    """
    logs = load_training_logs(log_dir)
    if logs is None:
        return
    
    rewards = logs['episode_rewards']
    episodes = list(range(len(rewards)))
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw rewards')
    plt.plot(episodes, smooth_curve(rewards, 50), color='blue', 
            linewidth=2, label='Smoothed (50 episodes)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('DQN Pong - Episode Rewards', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main entry point for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DQN training progress')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory containing training logs')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save plot')
    parser.add_argument('--simple', action='store_true',
                       help='Show only reward plot')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DQN PONG - TRAINING VISUALIZATION")
    print("=" * 80)
    
    if args.simple:
        plot_rewards_only(args.log_dir, args.save)
    else:
        plot_training_progress(args.log_dir, args.save)


if __name__ == "__main__":
    main()
