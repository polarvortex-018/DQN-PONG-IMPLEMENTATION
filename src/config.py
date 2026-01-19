"""
Configuration file for DQN Pong Agent

This file contains all hyperparameters and settings for training.
Each parameter is explained so you understand what it does and how to tune it.
"""

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

# Environment name - the Atari Pong game
ENV_NAME = "ALE/Pong-v5"

# Frame preprocessing settings
FRAME_WIDTH = 84          # Resize frames to 84x84 (standard for Atari DQN)
FRAME_HEIGHT = 84         # Square frames work well with CNNs
FRAME_STACK = 4           # Stack 4 frames to capture motion/velocity

# Why stack frames?
# A single frame doesn't show which direction the ball is moving!
# By stacking 4 consecutive frames, the network can "see" motion.

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

# Convolutional layers configuration
# Format: (filters, kernel_size, stride)
CONV_LAYERS = [
    (32, 8, 4),   # First layer: detect basic features (edges, corners)
    (64, 4, 2),   # Second layer: combine features (shapes, patterns)
    (64, 3, 1),   # Third layer: high-level features (ball, paddle positions)
]

# Fully connected layer size
FC_SIZE = 512             # 512 neurons to combine all spatial information

# Number of possible actions in Pong
# 0: NOOP (do nothing)
# 1: FIRE (start game)
# 2: RIGHT (move paddle up)
# 3: LEFT (move paddle down)
# 4: RIGHTFIRE (move up and start)
# 5: LEFTFIRE (move down and start)
# We'll simplify to just 3 actions: stay, up, down
NUM_ACTIONS = 6           # Will be set dynamically from environment

# ============================================================================
# DQN HYPERPARAMETERS
# ============================================================================

# Learning rate - how much to update network weights each step
LEARNING_RATE = 0.0005    # Increased from 0.00025 - balance between stability and progress
                          # Large value = fast but unstable learning
                          # 0.00025 is proven to work well for Atari

# Discount factor (gamma) - how much we value future rewards
GAMMA = 0.99              # 0.99 = care about long-term rewards
                          # 0.0 = only care about immediate rewards
                          # Higher gamma = more strategic play

# Batch size - number of experiences to train on at once
BATCH_SIZE = 32           # Stable gradient updates (was 64, too noisy)
                          # - Speed (larger = faster but more memory)
                          # - Stability (smaller = more stable updates)

# Replay buffer size - how many past experiences to remember
REPLAY_BUFFER_SIZE = 80000  # Proven stable size with uint8 storage
                             # Larger = more diverse training data
                             # Smaller = faster but less diverse

# Minimum experiences before training starts
MIN_REPLAY_SIZE = 20000   # 25% of buffer - ensures diversity
                          # This ensures diverse initial data

# Target network update frequency
TARGET_UPDATE_FREQ = 10000  # Conservative - more stable learning
                            # More frequent = less stable
                            # Less frequent = slower to adapt

# ============================================================================
# EXPLORATION SETTINGS (Epsilon-Greedy)
# ============================================================================

# Epsilon controls exploration vs exploitation
EPSILON_START = 1.0       # Start with 100% random actions (full exploration)
EPSILON_END = 0.01        # End with 1% random actions (maximum exploitation)
EPSILON_DECAY = 0.995     # Slower decay - better exploration before exploiting

# Epsilon decay schedule example:
# Episode 1: ε = 1.0 (100% random)
# Episode 100: ε ≈ 0.61 (61% random)
# Episode 500: ε ≈ 0.08 (8% random)
# Episode 1000+: ε = 0.1 (10% random, stays here)

# Alternative: Linear decay
USE_LINEAR_EPSILON_DECAY = False
EPSILON_DECAY_EPISODES = 1000  # If linear, decay over this many episodes

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

# Number of episodes to train
MAX_EPISODES = 3000       # 3000 episodes should be enough to master Pong
                          # Each episode = one game (until score reaches 21)

# Maximum steps per episode (safety limit)
MAX_STEPS_PER_EPISODE = 10000  # Prevent infinite episodes

# How often to save the model
SAVE_FREQ = 50           # Save checkpoint every 50 episodes

# How often to evaluate the agent (without exploration)
EVAL_FREQ = 50            # Evaluate every 50 episodes
EVAL_EPISODES = 5         # Run 5 evaluation games each time

# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

# How often to print training progress
PRINT_FREQ = 10           # Print stats every 10 episodes

# How often to save training metrics
LOG_FREQ = 1              # Log every episode

# TensorBoard logging
USE_TENSORBOARD = True    # Enable TensorBoard for visualization
TENSORBOARD_DIR = "logs/tensorboard"

# Video recording
RECORD_VIDEO = True       # Record videos during evaluation
VIDEO_FREQ = 100          # Record every 100 episodes
VIDEO_DIR = "videos"

# ============================================================================
# FILE PATHS
# ============================================================================

# Directory to save model checkpoints
CHECKPOINT_DIR = "checkpoints"

# Directory to save training logs
LOG_DIR = "logs"

# Best model filename
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.weights.h5"

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Frame skip - repeat each action this many times
FRAME_SKIP = 4            # Agent sees every 4th frame, actions repeat
                          # This speeds up training 4x!
                          # Pong doesn't need 60 FPS decision-making

# Enable GPU memory growth (prevents TensorFlow from hogging all GPU memory)
GPU_MEMORY_GROWTH = True

# Mixed precision training (faster on RTX 3050)
USE_MIXED_PRECISION = False  # Re-enabled for 16GB RAM system

# Number of parallel environments (advanced)
NUM_ENVS = 1              # Start with 1, can increase for faster training

# ============================================================================
# REWARD SHAPING (Optional)
# ============================================================================

# Pong gives sparse rewards: +1 for scoring, -1 for getting scored on
# We can add small rewards to help learning

USE_REWARD_SHAPING = True  # Set to True to enable

# Small reward for hitting the ball
BALL_HIT_REWARD = 0.1

# Small penalty for missing the ball
BALL_MISS_PENALTY = -0.1

# ============================================================================
# DEBUGGING AND TESTING
# ============================================================================

# Render environment during training (slow, only for debugging)
RENDER_TRAINING = False

# Seed for reproducibility
RANDOM_SEED = 42

# Verbose logging
VERBOSE = True

# ============================================================================
# ADVANCED SETTINGS (Don't change unless you know what you're doing!)
# ============================================================================

# Gradient clipping - prevents exploding gradients
CLIP_GRADIENTS = True
GRADIENT_CLIP_NORM = 10.0

# Huber loss delta (smoother than MSE for large errors)
HUBER_LOSS_DELTA = 1.0

# Optimizer
OPTIMIZER = "Adam"        # Options: "Adam", "RMSprop", "SGD"

# Learning rate schedule
USE_LR_SCHEDULE = False
LR_DECAY_RATE = 0.96
LR_DECAY_STEPS = 100000

# Double DQN (improved target calculation)
USE_DOUBLE_DQN = True     # Recommended: reduces overestimation

# Dueling DQN (separate value and advantage streams)
USE_DUELING_DQN = False   # Advanced: can improve performance

# Prioritized Experience Replay
USE_PRIORITIZED_REPLAY = False  # Advanced: sample important experiences more

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_epsilon(episode):
    """
    Calculate epsilon for given episode number.
    
    Args:
        episode: Current episode number
        
    Returns:
        Epsilon value (exploration rate)
    """
    if USE_LINEAR_EPSILON_DECAY:
        # Linear decay
        epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPSILON_DECAY_EPISODES)
        return max(EPSILON_END, epsilon)
    else:
        # Exponential decay
        epsilon = EPSILON_START * (EPSILON_DECAY ** episode)
        return max(EPSILON_END, epsilon)


def print_config():
    """Print all configuration settings."""
    print("=" * 80)
    print("DQN PONG AGENT CONFIGURATION")
    print("=" * 80)
    print(f"\nEnvironment: {ENV_NAME}")
    print(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}x{FRAME_STACK}")
    print(f"\nNeural Network:")
    print(f"  Conv layers: {CONV_LAYERS}")
    print(f"  FC size: {FC_SIZE}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Replay buffer: {REPLAY_BUFFER_SIZE}")
    print(f"  Target update freq: {TARGET_UPDATE_FREQ}")
    print(f"\nExploration:")
    print(f"  Epsilon start: {EPSILON_START}")
    print(f"  Epsilon end: {EPSILON_END}")
    print(f"  Epsilon decay: {EPSILON_DECAY}")
    print(f"\nTraining:")
    print(f"  Max episodes: {MAX_EPISODES}")
    print(f"  Frame skip: {FRAME_SKIP}")
    print(f"  GPU memory growth: {GPU_MEMORY_GROWTH}")
    print(f"  Mixed precision: {USE_MIXED_PRECISION}")
    print(f"  Double DQN: {USE_DOUBLE_DQN}")
    print(f"  Use Reward Shaping: {USE_REWARD_SHAPING}")
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    print_config()
    
    # Test epsilon decay
    print("\nEpsilon Decay Schedule:")
    print("-" * 40)
    for ep in [0, 100, 500, 1000, 2000, 3000]:
        print(f"Episode {ep:4d}: ε = {get_epsilon(ep):.4f}")
