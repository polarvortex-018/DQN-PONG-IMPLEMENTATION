"""
DQN Model - Deep Q-Network Architecture

This module implements the neural network that approximates Q-values.

Architecture:
    Input: 84x84x4 stacked frames
    → Conv Layer 1: 32 filters, 8x8, stride 4
    → Conv Layer 2: 64 filters, 4x4, stride 2
    → Conv Layer 3: 64 filters, 3x3, stride 1
    → Flatten
    → Dense: 512 neurons
    → Output: Q-values for each action

This is the same architecture from the original DQN paper!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def create_dqn_model(input_shape, num_actions):
    """
    Create the DQN neural network.
    
    This network takes game frames as input and outputs Q-values
    for each possible action.
    
    Args:
        input_shape: Shape of input (84, 84, 4)
        num_actions: Number of possible actions
        
    Returns:
        Keras model
    """
    # Ensure num_actions is a Python int (not numpy int)
    num_actions = int(num_actions)
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='state_input')
    
    # Convolutional layers
    # These extract visual features from the game frames
    
    # Layer 1: Detect basic features (edges, corners)
    # 32 filters of size 8x8, stride 4 (reduces 84x84 to 20x20)
    x = layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=4,
        activation='relu',
        name='conv1'
    )(inputs)
    
    # Layer 2: Combine features into patterns
    # 64 filters of size 4x4, stride 2 (reduces 20x20 to 9x9)
    x = layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        activation='relu',
        name='conv2'
    )(x)
    
    # Layer 3: High-level features (ball position, paddle position)
    # 64 filters of size 3x3, stride 1 (reduces 9x9 to 7x7)
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation='relu',
        name='conv3'
    )(x)
    
    # Flatten the 3D output to 1D
    # 7x7x64 = 3136 features
    x = layers.Flatten(name='flatten')(x)
    
    # Fully connected layer
    # Combines all spatial information
    x = layers.Dense(
        units=config.FC_SIZE,
        activation='relu',
        name='fc1'
    )(x)
    
    # Output layer: Q-value for each action
    # No activation (linear output)
    outputs = layers.Dense(
        units=num_actions,
        activation='linear',
        name='q_values'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='DQN')
    
    return model


def create_dueling_dqn_model(input_shape, num_actions):
    """
    Create a Dueling DQN architecture (advanced).
    
    Dueling DQN separates the Q-value into:
    - Value: How good is this state?
    - Advantage: How much better is each action?
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    
    This often learns faster because it can learn state values
    independently from action advantages.
    
    Args:
        input_shape: Shape of input (84, 84, 4)
        num_actions: Number of possible actions
        
    Returns:
        Keras model
    """
    # Ensure num_actions is a Python int (not numpy int)
    num_actions = int(num_actions)
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='state_input')
    
    # Shared convolutional layers (same as regular DQN)
    x = layers.Conv2D(32, 8, 4, activation='relu', name='conv1')(inputs)
    x = layers.Conv2D(64, 4, 2, activation='relu', name='conv2')(x)
    x = layers.Conv2D(64, 3, 1, activation='relu', name='conv3')(x)
    x = layers.Flatten(name='flatten')(x)
    
    # Split into two streams
    
    # Value stream: V(s) - how good is this state?
    value_stream = layers.Dense(256, activation='relu', name='value_fc')(x)
    value = layers.Dense(1, activation='linear', name='value')(value_stream)
    
    # Advantage stream: A(s,a) - how good is each action?
    advantage_stream = layers.Dense(256, activation='relu', name='advantage_fc')(x)
    advantage = layers.Dense(num_actions, activation='linear', name='advantage')(advantage_stream)
    
    # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    # Subtracting mean makes the advantage centered at 0
    mean_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - mean_advantage)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=q_values, name='Dueling_DQN')
    
    return model


class DQNAgent:
    """
    DQN Agent with main and target networks.
    
    This class manages:
    - Main network (updated every step)
    - Target network (updated periodically)
    - Training logic
    """
    
    def __init__(self, state_shape, num_actions, learning_rate=None):
        """
        Initialize DQN agent.
        
        Args:
            state_shape: Shape of state (84, 84, 4)
            num_actions: Number of possible actions
            learning_rate: Learning rate (uses config if None)
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate or config.LEARNING_RATE
        
        # Create main and target networks
        if config.USE_DUELING_DQN:
            self.main_network = create_dueling_dqn_model(state_shape, num_actions)
            self.target_network = create_dueling_dqn_model(state_shape, num_actions)
            print("Using Dueling DQN architecture")
        else:
            self.main_network = create_dqn_model(state_shape, num_actions)
            self.target_network = create_dqn_model(state_shape, num_actions)
            print("Using standard DQN architecture")
        
        # Initialize target network with same weights as main
        self.update_target_network()
        
        # Set up optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Loss function: Huber loss (smoother than MSE for large errors)
        self.loss_fn = keras.losses.Huber(delta=config.HUBER_LOSS_DELTA)
        
        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        
        print(f"DQN Agent initialized")
        print(f"  State shape: {state_shape}")
        print(f"  Number of actions: {num_actions}")
        print(f"  Learning rate: {self.learning_rate}")
        
        # Print model summary
        print("\nMain Network Architecture:")
        self.main_network.summary()
    
    def predict(self, state, use_target=False):
        """
        Predict Q-values for a state.
        
        Args:
            state: State to predict for (can be single state or batch)
            use_target: Whether to use target network
            
        Returns:
            Q-values for each action
        """
        # Add batch dimension if single state
        if len(state.shape) == 3:
            state = tf.expand_dims(state, axis=0)
        
        # Choose network
        network = self.target_network if use_target else self.main_network
        
        # Predict
        q_values = network(state, training=False)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (0 = always exploit, 1 = always explore)
            
        Returns:
            Action to take
        """
        # Explore: random action
        if tf.random.uniform(()) < epsilon:
            return tf.random.uniform((), 0, self.num_actions, dtype=tf.int32).numpy()
        
        # Exploit: best action according to Q-values
        q_values = self.predict(state)
        return tf.argmax(q_values[0]).numpy()
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Perform one training step.
        
        This is where the learning happens!
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Loss value
        """
        # Calculate target Q-values
        if config.USE_DOUBLE_DQN:
            # Double DQN: use main network to select action,
            # target network to evaluate it
            # This reduces overestimation of Q-values
            
            # Get best actions from main network
            next_q_main = self.main_network(next_states, training=False)
            best_actions = tf.argmax(next_q_main, axis=1)
            
            # Evaluate those actions with target network
            next_q_target = self.target_network(next_states, training=False)
            next_q_values = tf.gather_nd(
                next_q_target,
                tf.stack([tf.range(tf.shape(best_actions)[0]), 
                         tf.cast(best_actions, tf.int32)], axis=1)
            )
        else:
            # Standard DQN: use max Q-value from target network
            next_q_target = self.target_network(next_states, training=False)
            next_q_values = tf.reduce_max(next_q_target, axis=1)
        
        # Calculate targets: r + γ * max Q(s', a') * (1 - done)
        # If done, target is just the reward (no future)
        targets = rewards + config.GAMMA * next_q_values * (1 - dones)
        
        # Train main network
        with tf.GradientTape() as tape:
            # Get current Q-values
            q_values = self.main_network(states, training=True)
            
            # Get Q-values for actions that were taken
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_values_for_actions = tf.gather_nd(q_values, action_indices)
            
            # Calculate loss
            loss = self.loss_fn(targets, q_values_for_actions)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        if config.CLIP_GRADIENTS:
            gradients, _ = tf.clip_by_global_norm(gradients, config.GRADIENT_CLIP_NORM)
        
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        
        return loss
    
    def update_target_network(self):
        """
        Copy weights from main network to target network.
        
        This should be called every TARGET_UPDATE_FREQ steps.
        """
        self.target_network.set_weights(self.main_network.get_weights())
    
    def save(self, filepath):
        """
        Save the main network weights.
        
        Args:
            filepath: Path to save weights
        """
        self.main_network.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load network weights.
        
        Args:
            filepath: Path to load weights from
        """
        self.main_network.load_weights(filepath)
        self.update_target_network()
        print(f"Model loaded from {filepath}")


def test_model():
    """
    Test the DQN model with dummy data.
    """
    print("Testing DQN Model...")
    print("=" * 60)
    
    # Create model
    state_shape = (84, 84, 4)
    num_actions = 6
    
    agent = DQNAgent(state_shape, num_actions)
    
    # Test prediction
    print("\nTesting prediction...")
    dummy_state = tf.random.normal((1, 84, 84, 4))
    q_values = agent.predict(dummy_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values.numpy()}")
    
    # Test action selection
    print("\nTesting action selection...")
    action = agent.get_action(dummy_state[0], epsilon=0.0)
    print(f"Best action (greedy): {action}")
    
    action = agent.get_action(dummy_state[0], epsilon=1.0)
    print(f"Random action (ε=1.0): {action}")
    
    # Test training step
    print("\nTesting training step...")
    batch_size = 32
    states = tf.random.normal((batch_size, 84, 84, 4))
    actions = tf.random.uniform((batch_size,), 0, num_actions, dtype=tf.int32)
    rewards = tf.random.uniform((batch_size,), -1, 1)
    next_states = tf.random.normal((batch_size, 84, 84, 4))
    dones = tf.random.uniform((batch_size,)) > 0.9
    
    loss = agent.train_step(states, actions, rewards, next_states, dones)
    print(f"Training loss: {loss.numpy():.4f}")
    
    print("\nTest complete!")


if __name__ == "__main__":
    # Run test when this file is executed directly
    test_model()
