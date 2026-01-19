"""
Test if Pong environment actually gives non-zero rewards when playing longer
"""

import sys
sys.path.insert(0, 'src')
from environment import PongEnvironment
import numpy as np

print("="*80)
print("PONG REWARD SPARSITY TEST")
print("="*80)

env = PongEnvironment()
state = env.reset()

total_steps = 0
reward_counts = {-1.0: 0, 0.0: 0, 1.0: 0}
all_rewards = []

print("\nPlaying 3 full episodes to see if we ever get non-zero rewards...")
print("(This might take a minute)\n")

for episode in range(3):
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    
    while not done and episode_steps < 2000:  # Max 2000 steps per episode
        # Take random action
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        # Track rewards
        if reward in reward_counts:
            reward_counts[reward] += 1
        all_rewards.append(reward)
        
        episode_reward += reward
        episode_steps += 1
        total_steps += 1
        state = next_state
    
    print(f"Episode {episode+1}:")
    print(f"  Steps: {episode_steps}")
    print(f"  Total reward: {episode_reward}")
    print(f"  Final score: {info.get('score', 'unknown')}")

print("\n" + "="*80)
print("REWARD STATISTICS")
print("="*80)
print(f"Total steps: {total_steps}")
print(f"\nReward distribution:")
print(f"  +1 (scored point): {reward_counts[1.0]} ({reward_counts[1.0]/total_steps*100:.2f}%)")
print(f"   0 (neutral):      {reward_counts[0.0]} ({reward_counts[0.0]/total_steps*100:.2f}%)")
print(f"  -1 (got scored on): {reward_counts[-1.0]} ({reward_counts[-1.0]/total_steps*100:.2f}%)")

non_zero_rewards = reward_counts[1.0] + reward_counts[-1.0]
print(f"\nNon-zero rewards: {non_zero_rewards}/{total_steps} ({non_zero_rewards/total_steps*100:.2f}%)")

if non_zero_rewards == 0:
    print("\n❌ CRITICAL: NO non-zero rewards in 3 episodes!")
    print("   This means the agent has NO learning signal!")
elif non_zero_rewards < 10:
    print("\n⚠️  WARNING: Very sparse rewards (< 10 in 3 episodes)")
    print("   Agent will learn very slowly")
else:
    print(f"\n✓ OK: {non_zero_rewards} scoring events occurred")
    print("  Rewards are sparse but present")

# Check unique rewards
unique_rewards = set(all_rewards)
print(f"\nUnique reward values seen: {sorted(unique_rewards)}")

env.close()
