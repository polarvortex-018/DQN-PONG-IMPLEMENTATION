#!/bin/bash
# Comprehensive Training Diagnostics
# Run this to diagnose training issues

echo "================================================================================"
echo "DQN PONG TRAINING DIAGNOSTICS"
echo "================================================================================"
echo ""

# 1. Check if training process is running
echo "[1] Training Process Status"
echo "----------------------------"
if ps aux | grep -q "[t]rain.py"; then
    echo "✓ Training process is RUNNING"
    ps aux | grep "[t]rain.py" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $4"%"}'
else
    echo "✗ Training process NOT running!"
fi
echo ""

# 2. Check current progress
echo "[2] Training Progress"
echo "---------------------"
if [ -f "logs/training_log.json" ]; then
    python3 << 'EOF'
import json
try:
    with open('logs/training_log.json', 'r') as f:
        data = json.load(f)
    
    episodes = data.get('episodes_completed', 0)
    total_steps = data.get('total_steps', 0)
    
    print(f"Episodes completed: {episodes}/3000")
    print(f"Total steps: {total_steps:,}")
    print(f"Progress: {episodes/30:.1f}%")
    
    if episodes > 0:
        avg_steps_per_ep = total_steps / episodes
        print(f"Avg steps/episode: {avg_steps_per_ep:.1f}")
except Exception as e:
    print(f"Error reading log: {e}")
EOF
else
    echo "✗ No training log found!"
fi
echo ""

# 3. Check if training has started
echo "[3] Training Status (Buffer Fill)"
echo "----------------------------------"
python3 << 'EOF'
import json
try:
    with open('logs/training_log.json', 'r') as f:
        data = json.load(f)
    
    episodes = data.get('episodes_completed', 0)
    total_steps = data.get('total_steps', 0)
    min_replay = 25000  # Your MIN_REPLAY_SIZE
    
    if total_steps >= min_replay:
        print(f"✓ Training STARTED (buffer: {total_steps:,} >= {min_replay:,})")
    else:
        remaining = min_replay - total_steps
        print(f"✗ Training NOT STARTED YET")
        print(f"  Buffer: {total_steps:,} / {min_replay:,}")
        print(f"  Need {remaining:,} more steps (~{remaining/85:.0f} more episodes)")
except Exception as e:
    print(f"Error: {e}")
EOF
echo ""

# 4. Check loss progression
echo "[4] Loss Analysis (CRITICAL)"
echo "----------------------------"
python3 << 'EOF'
import json
import numpy as np
try:
    with open('logs/training_log.json', 'r') as f:
        data = json.load(f)
    
    losses = data.get('losses', [])
    
    if len(losses) == 0:
        print("✗ NO LOSS DATA - Training hasn't started!")
    elif len(losses) < 50:
        print(f"⚠ Only {len(losses)} loss values - too early to judge")
        print(f"  Current loss: {losses[-1]:.4f}")
    else:
        first_50 = np.mean(losses[:50])
        last_50 = np.mean(losses[-50:])
        trend = last_50 - first_50
        
        print(f"First 50 episodes: {first_50:.4f}")
        print(f"Last 50 episodes:  {last_50:.4f}")
        print(f"Change: {trend:+.4f}")
        
        if trend < -0.001:
            print("✓ Loss is DECREASING - Learning is happening!")
        elif abs(trend) < 0.001:
            print("✗ Loss is FLAT - NO learning (stuck)")
        else:
            print("✗ Loss is INCREASING - Divergence!")
            
except Exception as e:
    print(f"Error: {e}")
EOF
echo ""

# 5. Check reward progression
echo "[5] Reward Analysis"
echo "-------------------"
python3 << 'EOF'
import json
import numpy as np
try:
    with open('logs/training_log.json', 'r') as f:
        data = json.load(f)
    
    rewards = data.get('episode_rewards', [])
    
    if len(rewards) < 100:
        recent = rewards[-10:] if len(rewards) >= 10 else rewards
        print(f"Episodes so far: {len(rewards)}")
        print(f"Last 10 avg: {np.mean(recent):.2f}")
    else:
        first_100 = np.mean(rewards[:100])
        last_100 = np.mean(rewards[-100:])
        improvement = last_100 - first_100
        
        print(f"First 100 episodes: {first_100:.2f}")
        print(f"Last 100 episodes:  {last_100:.2f}")
        print(f"Improvement: {improvement:+.2f}")
        
        if improvement > 2:
            print("✓ Rewards IMPROVING - Agent learning!")
        elif abs(improvement) < 2:
            print("✗ Rewards FLAT - No improvement")
        else:
            print("✗ Rewards WORSE - Agent degrading")
            
except Exception as e:
    print(f"Error: {e}")
EOF
echo ""

# 6. Memory usage
echo "[6] System Resources"
echo "--------------------"
free -h | grep "Mem:" | awk '{print "RAM: "$3" used / "$2" total | "$7" available"}'
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F',' '{print "GPU: "$1" util | "$2" / "$3}'
fi
echo ""

# 7. Check for errors in log
echo "[7] Recent Errors in Training Log"
echo "-----------------------------------"
if [ -f training_logs/training_*.log ]; then
    latest_log=$(ls -t training_logs/training_*.log | head -1)
    echo "Checking: $latest_log"
    if grep -i "error\|exception\|traceback" "$latest_log" | tail -5; then
        echo ""
    else
        echo "✓ No recent errors found"
    fi
else
    echo "No training log file found"
fi
echo ""

# 8. Configuration check
echo "[8] Current Configuration"
echo "-------------------------"
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
import config

print(f"Learning Rate: {config.LEARNING_RATE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Buffer Size: {config.REPLAY_BUFFER_SIZE:,}")
print(f"Min Replay Size: {config.MIN_REPLAY_SIZE:,}")
print(f"Epsilon Decay: {config.EPSILON_DECAY}")
EOF
echo ""

echo "================================================================================"
echo "DIAGNOSIS COMPLETE"
echo "================================================================================"
