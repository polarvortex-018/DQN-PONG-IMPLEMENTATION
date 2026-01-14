"""
Replay Buffer for Experience Replay

This module implements the replay buffer (also called replay memory),
which stores past experiences and samples random batches for training.

Why do we need this?
- Breaks correlation between consecutive experiences
- Allows reusing experiences multiple times
- Stabilizes training by providing diverse batches

Think of it as a "memory" where the agent stores what it has experienced,
then randomly recalls those memories to learn from them.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Circular buffer to store and sample experiences.
    
    Each experience is a tuple: (state, action, reward, next_state, done)
    
    The buffer has a maximum size. When full, old experiences are
    automatically removed to make room for new ones.
    """
    
    def __init__(self, max_size):
        """
        Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
        print(f"Replay buffer initialized with max size: {max_size:,}")
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state (84, 84, 4)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state after action (84, 84, 4)
            done: Whether episode ended (bool)
        """
        # Store as tuple
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a random batch of experiences.
        
        This is where the magic happens! By sampling randomly,
        we break the correlation between consecutive experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of numpy arrays:
                states: (batch_size, 84, 84, 4)
                actions: (batch_size,)
                rewards: (batch_size,)
                next_states: (batch_size, 84, 84, 4)
                dones: (batch_size,)
        """
        # Sample random experiences
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch into separate arrays
        # zip(*batch) transposes the list of tuples
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays for TensorFlow
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get current size of buffer.
        
        Returns:
            Number of experiences currently stored
        """
        return len(self.buffer)
    
    def is_ready(self, min_size):
        """
        Check if buffer has enough experiences to start training.
        
        Args:
            min_size: Minimum number of experiences needed
            
        Returns:
            True if buffer has at least min_size experiences
        """
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def get_stats(self):
        """
        Get statistics about the buffer contents.
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'num_episodes': 0
            }
        
        # Extract rewards and dones from buffer
        rewards = [exp[2] for exp in self.buffer]
        dones = [exp[4] for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'utilization': len(self.buffer) / self.max_size,
            'avg_reward': np.mean(rewards),
            'num_episodes': sum(dones)
        }
