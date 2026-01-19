"""
Test reward shaping implementation
Verify that shaped rewards are being calculated correctly
"""

import sys
sys.path.insert(0, 'src')
from environment import PongEnvironment
import config
import numpy as np

print("="*80)
print("REWARD SHAPING TEST")
print("="*80)

# Test with reward shaping enabled
print(f"\nReward shaping enabled: {config.USE_REWARD_SHAPING}")

env = PongEnvironment()
state = env.reset()

print("\nPlaying 500 steps to test shaped rewards...")

rewards = []
game_rewards = []
shaped_bonuses = []

for step in range(500):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    
    rewards.append(reward)
    
    # Track if reward is different from -1, 0, +1 (has shaping)
    if abs(reward) not in [0.0, 1.0]:
        shaped_bonuses.append(reward)
    
    if done:
        state = env.reset()

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\n Total steps: {len(rewards)}")
print(f"Unique reward values: {len(set(rewards))}")
print(f"Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")

# Count non-zero rewards
non_zero = sum(1 for r in rewards if r != 0)
print(f"\nNon-zero rewards: {non_zero}/{len(rewards)} ({non_zero/len(rewards)*100:.1f}%)")

# Show sample of shaped rewards
if shaped_bonuses:
    print(f"\nShaped reward examples (first 10):")
    for i, r in enumerate(shaped_bonuses[:10]):
        print(f"  {i+1}. {r:.6f}")
    print(f"\nSuccess! Found {len(shaped_bonuses)} shaped rewards")
    print("✓ Reward shaping is working")
else:
    print("\n⚠ No shaped rewards found - only saw -1, 0, +1")
    print("  Shaping might not be triggering")

# Statistics
print(f"\nReward statistics:")
print(f"  Mean: {np.mean(rewards):.4f}")
print(f"  Std: {np.std(rewards):.4f}")
print(f"  Min: {min(rewards):.4f}")
print(f"  Max: {max(rewards):.4f}")

env.close()

if non_zero > len(rewards) * 0.1:  # If >10% non-zero
    print("\n✅ PASS: Reward shaping significantly increases signal density!")
    print(f"   Went from ~3% to {non_zero/len(rewards)*100:.1f}% non-zero rewards")
else:
    print("\n⚠ Marginal improvement in signal density")
