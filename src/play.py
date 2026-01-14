"""
Play Script - Watch Your Trained Agent Play Pong!

This script loads a trained model and lets you watch it play.
You can also record videos of the gameplay.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import imageio
from datetime import datetime

import config
from environment import PongEnvironment
from dqn_model import DQNAgent


def play_episode(env, agent, render=True, record=False):
    """
    Play one episode with the trained agent.
    
    Args:
        env: Environment
        agent: Trained agent
        render: Whether to display the game
        record: Whether to record frames
        
    Returns:
        episode_reward: Total reward
        frames: List of frames (if recording)
    """
    state = env.reset()
    done = False
    episode_reward = 0
    frames = [] if record else None
    
    while not done:
        # Get action (greedy, no exploration)
        action = agent.get_action(state, epsilon=0.0)
        
        # Take action
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Record frame if needed
        if record:
            # Get RGB frame from environment
            frame = env.env.render()
            if frame is not None:
                frames.append(frame)
    
    return episode_reward, frames


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch trained DQN agent play Pong')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to play')
    parser.add_argument('--record', action='store_true',
                       help='Record video of gameplay')
    parser.add_argument('--no-render', action='store_true',
                       help='Do not display the game')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DQN PONG AGENT - PLAY MODE")
    print("=" * 80)
    
    # Create environment
    #render_mode = None if args.no_render else "rgb_array"
    render_mode = "human" if not args.no_render else None
    env = PongEnvironment(render_mode=render_mode)
    
    # Create agent
    agent = DQNAgent(env.get_state_shape(), env.get_num_actions())
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    agent.load(args.checkpoint)
    print(f"\nModel loaded from: {args.checkpoint}")
    print(f"Playing {args.episodes} episodes...\n")
    
    # Play episodes
    rewards = []
    all_frames = []
    
    for i in range(args.episodes):
        print(f"Episode {i+1}/{args.episodes}...", end=" ")
        
        reward, frames = play_episode(env, agent, 
                                      render=not args.no_render,
                                      record=args.record)
        rewards.append(reward)
        
        if args.record and frames:
            all_frames.extend(frames)
        
        print(f"Reward: {reward:.0f}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Best reward: {np.max(rewards):.0f}")
    print(f"Worst reward: {np.min(rewards):.0f}")
    print(f"Win rate: {sum(1 for r in rewards if r > 0) / len(rewards) * 100:.1f}%")
    
    # Save video if recorded
    if args.record and all_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(config.VIDEO_DIR, f"pong_gameplay_{timestamp}.mp4")
        
        print(f"\nSaving video to: {video_path}")
        imageio.mimsave(video_path, all_frames, fps=30)
        print("Video saved!")
    
    env.close()


if __name__ == "__main__":
    main()
