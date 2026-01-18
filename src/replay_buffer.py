"""
Memory-Efficient Replay Buffer for Atari DQN

This implementation stores ONLY single uint8 frames (84x84) and dynamically
reconstructs 4-frame stacks at sampling time. This reduces memory usage by 32x
compared to storing pre-stacked float32 states.

Memory per experience:
- OLD: (84×84×4 float32) × 2 states = ~225 KB per experience
- NEW: (84×84 uint8) × 1 frame = ~7 KB per experience
- Savings: 32x less memory!
"""

import numpy as np


class ReplayBuffer:
    """
    Memory-efficient circular buffer for Atari DQN.
    
    Stores only single uint8 frames and reconstructs stacked states on-the-fly.
    This dramatically reduces RAM usage while maintaining full compatibility
    with the existing training loop.
    """
    
    def __init__(self, capacity, frame_stack=4):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            frame_stack: Number of frames to stack (default: 4 for Atari)
        """
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.size = 0
        self.pos = 0
        
        # Pre-allocate NumPy arrays for maximum efficiency
        # Store frames as uint8 (0-255) to save memory
        self.frames = np.zeros((capacity, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        print(f"Replay buffer initialized with max size: {capacity:,}")
        print(f"Memory-efficient mode: storing uint8 frames only")
        
        # Calculate memory savings
        old_memory_mb = (capacity * 84 * 84 * 4 * 4 * 2) / (1024**2)  # float32, 2 states
        new_memory_mb = (capacity * 84 * 84) / (1024**2)  # uint8, 1 frame
        print(f"Memory usage: ~{new_memory_mb:.1f}MB (vs {old_memory_mb:.1f}MB with old method)")
    
    def add(self, frame_uint8, action, reward, done):
        """
        Add a single-frame transition to the buffer.
        
        Args:
            frame_uint8: Single grayscale frame (84, 84) as uint8 [0-255]
            action: Action taken (int)
            reward: Reward received (float)
            done: Whether episode ended (bool)
        """
        assert frame_uint8.shape == (84, 84), f"Expected (84, 84) frame, got {frame_uint8.shape}"
        assert frame_uint8.dtype == np.uint8, f"Expected uint8 frame, got {frame_uint8.dtype}"
        
        # Store in circular buffer
        self.frames[self.pos] = frame_uint8
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
        # Update position (circular)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _get_stacked_state(self, idx):
        """
        Reconstruct a 4-frame stacked state for a given index.
        
        Returns frames [idx-3, idx-2, idx-1, idx] stacked along axis 2.
        Handles episode boundaries correctly - won't stack across episodes.
        
        Args:
            idx: Index in the buffer
            
        Returns:
            Stacked state (84, 84, 4) as float32 normalized to [0, 1]
        """
        indices = []
        
        # Build indices going backwards from idx
        for i in range(self.frame_stack):
            frame_idx = idx - (self.frame_stack - 1 - i)
            
            # Check if this crosses episode boundary
            if frame_idx < 0:
                # Before buffer start, repeat first available frame
                indices.append(max(0, idx - (self.frame_stack - 1)))
            elif frame_idx > 0 and self.dones[frame_idx - 1]:
                # Episode boundary detected, start fresh from here
                indices = [frame_idx]
            else:
                indices.append(frame_idx)
        
        # Ensure we have frame_stack indices
        while len(indices) < self.frame_stack:
            indices.insert(0, indices[0])
        
        # Stack frames along last axis
        stacked = np.stack([self.frames[i] for i in indices], axis=-1)
        
        # Convert to float32 and normalize to [0, 1]
        return stacked.astype(np.float32) / 255.0
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Reconstructs stacked states and next_states on-the-fly from stored frames.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of numpy arrays compatible with existing training loop:
                states: (batch_size, 84, 84, 4) as float32 [0, 1]
                actions: (batch_size,) as int32
                rewards: (batch_size,) as float32
                next_states: (batch_size, 84, 84, 4) as float32 [0, 1]
                dones: (batch_size,) as float32
        """
        # Build list of valid indices
        # Valid if: has enough history, not terminal, has next frame
        valid_indices = []
        
        for i in range(self.frame_stack, self.size - 1):
            # Don't sample terminal states (can't have meaningful next_state)
            if not self.dones[i]:
                valid_indices.append(i)
        
        # Handle edge case: not enough valid indices
        if len(valid_indices) < batch_size:
            if len(valid_indices) == 0:
                # Very early in training, use what we have
                valid_indices = list(range(self.frame_stack, max(self.frame_stack + 1, self.size)))
            batch_size = min(batch_size, len(valid_indices))
        
        # Sample random indices
        sampled_indices = np.random.choice(valid_indices, batch_size, replace=False)
        
        # Reconstruct states and next_states
        states = np.array([self._get_stacked_state(idx) for idx in sampled_indices])
        next_states = np.array([self._get_stacked_state(idx + 1) for idx in sampled_indices])
        
        # Extract actions, rewards, dones
        actions = self.actions[sampled_indices]
        rewards = self.rewards[sampled_indices]
        dones = self.dones[sampled_indices].astype(np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Get current number of transitions in buffer."""
        return self.size
    
    def is_ready(self, min_size):
        """
        Check if buffer has enough experiences to start training.
        
        Args:
            min_size: Minimum number of experiences needed
            
        Returns:
            True if buffer has at least min_size experiences
        """
        return self.size >= min_size
    
    def clear(self):
        """Reset buffer to empty state."""
        self.pos = 0
        self.size = 0
        self.frames.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(False)
    
    def get_stats(self):
        """
        Get statistics about buffer contents.
        
        Returns:
            Dictionary with buffer statistics
        """
        if self.size == 0:
            return {
                'size': 0,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'num_episodes': 0
            }
        
        valid_rewards = self.rewards[:self.size]
        valid_dones = self.dones[:self.size]
        
        return {
            'size': self.size,
            'utilization': self.size / self.capacity,
            'avg_reward': float(np.mean(valid_rewards)),
            'num_episodes': int(np.sum(valid_dones))
        }
