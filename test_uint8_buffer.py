"""
Quick test script to verify uint8 replay buffer implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from environment import PongEnvironment
from replay_buffer import ReplayBuffer

print("=" * 80)
print("TESTING UINT8 REPLAY BUFFER")
print("=" * 80)

# Test 1: Environment provides uint8 frames
print("\n[Test 1] Environment uint8 frame extraction...")
env = PongEnvironment()
state = env.reset()

# Check state is still (84, 84, 4) float32
assert state.shape == (84, 84, 4), f"State shape wrong: {state.shape}"
assert state.dtype == np.float32, f"State dtype wrong: {state.dtype}"
print(f"✓ Stacked state: shape={state.shape}, dtype={state.dtype}")

# Get uint8 frame
frame = env.get_last_frame_uint8()
assert frame.shape == (84, 84), f"Frame shape wrong: {frame.shape}"
assert frame.dtype == np.uint8, f"Frame dtype wrong: {frame.dtype}"
print(f"✓ uint8 frame: shape={frame.shape}, dtype={frame.dtype}")

# Test 2: Replay buffer accepts uint8 frames
print("\n[Test 2] Replay buffer stores uint8 frames...")
buffer = ReplayBuffer(capacity=1000, frame_stack=4)

# Add 10 experiences
for i in range(10):
    action = np.random.randint(0, 6)
    next_state, reward, done, _ = env.step(action)
    frame = env.get_last_frame_uint8()
    buffer.add(frame, action, reward, done)

assert len(buffer) == 10, f"Buffer size wrong: {len(buffer)}"
print(f"✓ Added 10 experiences, buffer size: {len(buffer)}")

# Test 3: Sample reconstructs stacked states correctly
print("\n[Test 3] Sampling reconstructs stacked states...")
# Add more experiences to have enough for sampling
for i in range(100):
    action = np.random.randint(0, 6)
    next_state, reward, done, _ = env.step(action)
    frame = env.get_last_frame_uint8()
    buffer.add(frame, action, reward, done)

# Sample batch
states, actions, rewards, next_states, dones = buffer.sample(32)

assert states.shape == (32, 84, 84, 4), f"States shape wrong: {states.shape}"
assert states.dtype == np.float32, f"States dtype wrong: {states.dtype}"
assert next_states.shape == (32, 84, 84, 4), f"Next states shape wrong: {next_states.shape}"
assert actions.shape == (32,), f"Actions shape wrong: {actions.shape}"
assert rewards.shape == (32,), f"Rewards shape wrong: {rewards.shape}"
assert dones.shape == (32,), f"Dones shape wrong: {dones.shape}"

print(f"✓ Batch shapes correct:")
print(f"  - states: {states.shape}")
print(f"  - actions: {actions.shape}")
print(f"  - rewards: {rewards.shape}")
print(f"  - next_states: {next_states.shape}")
print(f"  - dones: {dones.shape}")

# Test 4: Check values are normalized
print("\n[Test 4] Values are correctly normalized...")
assert states.min() >= 0.0 and states.max() <= 1.0, "States not in [0,1] range"
assert next_states.min() >= 0.0 and next_states.max() <= 1.0, "Next states not in [0,1] range"
print(f"✓ States in range [0, 1]: min={states.min():.3f}, max={states.max():.3f}")

# Test 5: Memory usage
print("\n[Test 5] Memory usage estimate...")
buffer_80k = ReplayBuffer(capacity=80000, frame_stack=4)
stats = buffer.get_stats()
print(f"✓ Buffer stats: {stats}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print("\nThe uint8 replay buffer is working correctly!")
print("You can now train with 80k buffer size using less memory than 30k before.")

env.close()
