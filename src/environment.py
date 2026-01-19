"""
Environment Wrapper for Atari Pong

This module wraps the Gymnasium Pong environment with preprocessing:
- Frame resizing (210x160 → 84x84)
- Grayscale conversion (RGB → grayscale)
- Frame stacking (4 frames to capture motion)
- Reward clipping (normalize rewards to -1, 0, +1)

Why preprocessing?
- Smaller frames = faster training, less memory
- Grayscale = color doesn't matter in Pong
- Frame stacking = network can see ball velocity
- Reward clipping = more stable learning
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import ale_py 
import config


class PongEnvironment:
    """
    Wrapper for Atari Pong environment with preprocessing.
    
    This class handles all the messy details of frame preprocessing
    so the agent just sees clean, ready-to-use observations.
    """
    
    def __init__(self, render_mode=None):
        """
        Initialize the Pong environment.
        
        Args:
            render_mode: "human" to display game, None for no display
        """
        # Create the base Gymnasium environment
        self.env = gym.make(
            config.ENV_NAME,
            render_mode=render_mode,
            frameskip=1,  # We'll handle frame skipping ourselves
        )
        
        # Frame stacking: keep last N frames to see motion
        self.frame_stack = deque(maxlen=config.FRAME_STACK)
        
        # Get action space info
        self.action_space = self.env.action_space
        self.num_actions = self.action_space.n
        
        # Track episode statistics
        self.episode_reward = 0
        self.episode_length = 0
        
        print(f"Environment initialized: {config.ENV_NAME}")
        print(f"Action space: {self.num_actions} actions")
        print(f"Observation shape: ({config.FRAME_HEIGHT}, {config.FRAME_WIDTH}, {config.FRAME_STACK})")
    
    def reset(self):
        """
        Reset environment to start a new episode.
        
        Returns:
            Initial state (stacked frames)
        """
        # Reset the environment
        obs, info = self.env.reset()
        
        # Reset statistics
        self.episode_reward = 0
        self.episode_length = 0
        
        # Preprocess the initial frame
        processed_frame = self._preprocess_frame(obs)
        
        # Fill frame stack with copies of first frame
        # (We don't have 4 frames yet, so duplicate the first one)
        for _ in range(config.FRAME_STACK):
            self.frame_stack.append(processed_frame)
        
        # Stack frames along last dimension: (84, 84, 4)
        state = np.stack(self.frame_stack, axis=-1).astype(np.float32)
        
        return state
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Action to take (0-5 for Pong)
            
        Returns:
            next_state: Next state after action
            reward: Reward received
            done: Whether episode ended
            info: Additional information
        """
        # Execute action with frame skipping
        # Repeat the action for FRAME_SKIP frames and accumulate rewards
        total_reward = 0
        done = False
        truncated = False
        
        for _ in range(config.FRAME_SKIP):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        # Preprocess the new frame
        processed_frame = self._preprocess_frame(obs)
        
        # Add to frame stack (automatically removes oldest frame)
        self.frame_stack.append(processed_frame)
        
        # Stack frames to create state
        next_state = np.stack(self.frame_stack, axis=-1).astype(np.float32)
        
        # Apply reward shaping if enabled
        if config.USE_REWARD_SHAPING:
            shaped_reward = self._calculate_shaped_reward(obs, total_reward, processed_frame)
        else:
            shaped_reward = total_reward
        
        # Update statistics
        self.episode_reward += total_reward  # Track game score
        self.episode_length += 1
        
        # Combine done and truncated
        done = done or truncated
        
        return next_state, shaped_reward, done, info
    
    def _calculate_shaped_reward(self, raw_obs, game_reward, processed_frame):
        """
        Calculate shaped reward with intermediate signals.
        
        Addresses sparse reward problem where 97% of frames have reward=0.
        Adds small rewards for:
        - Ball getting closer to opponent (+)
        - Ball being near our paddle when we could hit it (+)
        - Ball near our goal (-)
        
        Args:
            raw_obs: Raw observation from environment (210x160 RGB)
            game_reward: Original game reward (+1/-1/0)
            processed_frame: Preprocessed grayscale frame
            
        Returns:
            shaped_reward: Game reward + shaping bonuses
        """
        shaped = game_reward
        
        # Detect ball and paddle positions using simple heuristics
        ball_x, ball_y = self._detect_ball(raw_obs)
        paddle_y = self._detect_paddle(raw_obs)
        
        if ball_x is not None and ball_y is not None:
            # Reward 1: Ball proximity to opponent side
            # Field is ~160 pixels wide, ball at x>80 is opponent side
            if ball_x > 80:
                proximity_bonus = 0.005 * ((ball_x - 80) / 80)  # 0 to 0.005
                shaped += proximity_bonus
            else:
                proximity_penalty = -0.005 * ((80 - ball_x) / 80)  # 0 to -0.005
                shaped += proximity_penalty
            
            # Reward 2: Ball near our paddle (potential to hit)
            if paddle_y is not None and ball_x < 40:  # Ball on our side
                vertical_dist = abs(ball_y - paddle_y)
                if vertical_dist < 20:  # Within hitting range
                    contact_bonus = 0.01 * (1 - vertical_dist / 20)  # 0.01 to 0
                    shaped += contact_bonus
        
        return shaped
    
    def _detect_ball(self, obs):
        """
        Detect ball position in raw Pong observation.
        
        Ball is a small white square (brightest object).
        
        Returns:
            (x, y) tuple or (None, None) if not found
        """
        # Convert to grayscale for simpler detection
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Ball is the brightest pixels
        threshold = 200
        bright_pixels = np.where(gray > threshold)
        
        if len(bright_pixels[0]) > 0:
            # Get center of bright region
            y_coords = bright_pixels[0]
            x_coords = bright_pixels[1]
            
            # Return centroid
            ball_y = int(np.mean(y_coords))
            ball_x = int(np.mean(x_coords))
            
            return ball_x, ball_y
        
        return None, None
    
    def _detect_paddle(self, obs):
        """
        Detect our paddle position (left side).
        
        Paddle is a vertical white rectangle on the left.
        
        Returns:
            y position or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Look for bright vertical line on left side
        left_region = gray[:, :30]  # Left 30 pixels
        bright = np.where(left_region > 200)
        
        if len(bright[0]) > 0:
            # Return vertical center of paddle
            paddle_y = int(np.mean(bright[0]))
            return paddle_y
        
        return None
    
    def _preprocess_frame(self, frame):
        """
        Preprocess a single frame.
        
        Steps:
        1. Convert RGB to grayscale (color doesn't help in Pong)
        2. Resize from 210x160 to 84x84 (smaller = faster)
        3. Normalize pixel values to [0, 1]
        
        Args:
            frame: Raw frame from environment (210, 160, 3)
            
        Returns:
            Preprocessed frame (84, 84)
        """
        # Convert to grayscale
        # We use a weighted average: 0.299*R + 0.587*G + 0.114*B
        # This matches human perception of brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84 using bilinear interpolation
        resized = cv2.resize(
            gray,
            (config.FRAME_WIDTH, config.FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize to [0, 1] range
        # Neural networks work better with normalized inputs
        normalized = resized / 255.0
        
        return normalized
    
    def get_state_shape(self):
        """
        Get the shape of preprocessed states.
        
        Returns:
            Tuple: (height, width, channels)
        """
        return (config.FRAME_HEIGHT, config.FRAME_WIDTH, config.FRAME_STACK)
    
    def get_num_actions(self):
        """
        Get number of possible actions.
        
        Returns:
            int: Number of actions
        """
        return self.num_actions
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def render(self):
        """Render the environment (if render_mode was set)."""
        # Rendering is handled automatically by Gymnasium
        pass
    
    def get_last_frame_uint8(self):
        """
        Get the most recent processed frame as uint8.
        
        This is used for memory-efficient replay buffer storage.
        The frame is converted from float32 [0,1] back to uint8 [0-255].
        
        Returns:
            numpy array: (84, 84) uint8 frame
        """
        if len(self.frame_stack) == 0:
            raise ValueError("No frames in stack yet - call reset() first")
        
        # Get last frame and convert float32 → uint8
        last_frame = self.frame_stack[-1]
        frame_uint8 = (last_frame * 255).astype(np.uint8)
        
        return frame_uint8


def test_environment():
    """
    Test the environment wrapper with random actions.
    
    This function demonstrates how to use the environment
    and lets you see Pong in action!
    """
    print("Testing Pong Environment...")
    print("=" * 60)
    
    # Create environment with rendering
    env = PongEnvironment(render_mode="human")
    
    # Print environment info
    print(f"State shape: {env.get_state_shape()}")
    print(f"Number of actions: {env.get_num_actions()}")
    print(f"Action meanings: {env.env.unwrapped.get_action_meanings()}")
    print("\nPlaying 3 episodes with random actions...")
    print("Close the window to stop.\n")
    
    # Play a few episodes
    for episode in range(3):
        state = env.reset()
        done = False
        step = 0
        
        print(f"Episode {episode + 1} started")
        
        while not done and step < 1000:
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Print when we score
            if reward != 0:
                print(f"  Step {step}: Reward = {reward}")
            
            state = next_state
            step += 1
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total reward: {env.episode_reward}")
        print(f"  Episode length: {env.episode_length}")
        print()
    
    env.close()
    print("Test complete!")


if __name__ == "__main__":
    # Run test when this file is executed directly
    test_environment()
