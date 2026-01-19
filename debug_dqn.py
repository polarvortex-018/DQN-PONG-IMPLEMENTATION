"""
DQN Implementation Debugging Script

This script adds diagnostic checks to verify each component of the DQN training loop.
Run this to identify where the learning failure is occurring.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import tensorflow as tf
from environment import PongEnvironment
from replay_buffer import ReplayBuffer
from dqn_model import DQNAgent
import config

print("="*80)
print("DQN IMPLEMENTATION DEBUGGING")
print("="*80)

# Test 1: Environment
print("\n[Test 1] Environment Functionality")
print("-" * 40)
env = PongEnvironment()
state = env.reset()
print(f"✓ Environment created")
print(f"  State shape: {state.shape}, dtype: {state.dtype}")
print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")

# Take 10 random actions
rewards = []
for _ in range(10):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    rewards.append(reward)
    if done:
        state = env.reset()
        
print(f"  Sample rewards: {rewards}")
print(f"  Unique rewards: {set(rewards)}")
if len(set(rewards)) == 1 and rewards[0] == 0:
    print("  ⚠ WARNING: All rewards are 0!")

# Test 2: Neural Network
print("\n[Test 2] Neural Network")
print("-" * 40)
agent = DQNAgent(env.get_state_shape(), env.get_num_actions())

# Test forward pass
q_values = agent.predict(state)
print(f"✓ Forward pass works")
print(f"  Q-values shape: {q_values.shape}")
print(f"  Q-values: {q_values.numpy()[0]}")
print(f"  Q-value range: [{q_values.numpy().min():.4f}, {q_values.numpy().max():.4f}]")

if np.allclose(q_values.numpy(), 0):
    print("  ⚠ WARNING: All Q-values are near zero!")

# Test 3: Replay Buffer
print("\n[Test 3] Replay Buffer")
print("-" * 40)
buffer = ReplayBuffer(1000)

# Add some experiences
state = env.reset()
for i in range(100):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    
    # Get uint8 frame for buffer
    frame = env.get_last_frame_uint8()
    buffer.add(frame, action, reward, done)
    
    if done:
        state = env.reset()
    else:
        state = next_state

print(f"✓ Buffer filled with {len(buffer)} experiences")

# Sample batch
states, actions, rewards, next_states, dones = buffer.sample(32)
print(f"  Sampled batch shapes:")
print(f"    states: {states.shape}")
print(f"    actions: {actions.shape}")
print(f"    rewards: {rewards.shape}")
print(f"    next_states: {next_states.shape}")
print(f"    dones: {dones.shape}")
print(f"  Sample rewards: {rewards[:5]}")
print(f"  Sample actions: {actions[:5]}")

# Test 4: Training Step
print("\n[Test 4] Training Step")
print("-" * 40)

# Get initial weights
initial_weights = agent.main_network.get_weights()[0].copy()
print(f"  Initial first layer weights (sample): {initial_weights.flat[:5]}")

# Perform one training step
loss = agent.train_step(states, actions, rewards, next_states, dones)
print(f"✓ Training step executed")
print(f"  Loss: {loss.numpy():.6f}")

# Check if weights changed
new_weights = agent.main_network.get_weights()[0]
weight_diff = np.abs(new_weights - initial_weights).max()
print(f"  Max weight change: {weight_diff:.10f}")

if weight_diff < 1e-10:
    print("  ❌ CRITICAL: Weights did NOT change!")
else:
    print(f"  ✓ Weights changed by {weight_diff:.10f}")

# Test 5: Gradient Flow
print("\n[Test 5] Gradient Flow")
print("-" * 40)

with tf.GradientTape() as tape:
    q_values = agent.main_network(states, training=True)
    action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    q_values_for_actions = tf.gather_nd(q_values, action_indices)
    
    # Simple loss
    target_q = rewards + config.GAMMA * tf.reduce_max(
        agent.target_network(next_states, training=False), axis=1
    ) * (1 - dones)
    
    loss = agent.loss_fn(target_q, q_values_for_actions)

gradients = tape.gradient(loss, agent.main_network.trainable_variables)

print(f"  Number of gradient tensors: {len(gradients)}")
print(f"  First layer gradient shape: {gradients[0].shape}")
print(f"  First layer gradient stats:")
print(f"    Mean: {tf.reduce_mean(tf.abs(gradients[0])).numpy():.10f}")
print(f"    Max: {tf.reduce_max(tf.abs(gradients[0])).numpy():.10f}")
print(f"    Min: {tf.reduce_min(tf.abs(gradients[0])).numpy():.10f}")

if tf.reduce_max(tf.abs(gradients[0])).numpy() < 1e-10:
    print("  ❌ CRITICAL: Gradients are near zero!")
else:
    print("  ✓ Gradients are non-zero")

# Test 6: Learning Rate Effect
print("\n[Test 6] Learning Rate Effect")
print("-" * 40)
print(f"  Current learning rate: {config.LEARNING_RATE}")
print(f"  Typical gradient magnitude: {tf.reduce_mean(tf.abs(gradients[0])).numpy():.10f}")
print(f"  Expected weight update: ~{config.LEARNING_RATE * tf.reduce_mean(tf.abs(gradients[0])).numpy():.10f}")

if config.LEARNING_RATE * tf.reduce_mean(tf.abs(gradients[0])).numpy() < 1e-8:
    print("  ⚠ WARNING: Learning rate * gradient is very small!")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\nCheck the warnings and critical issues above.")
print("The most likely culprits are marked with ❌")
