# RL Pong Training - Complete Command Reference

## 🚀 Quick Start (Every Time You Open Terminal)

```bash
# 1. Navigate to project directory
cd "/mnt/c/Users/renbou/Desktop/Most of the Good Stuff/projects/REINFORCEMENT LEARNING PONG"

# 2. Activate virtual environment
source venv-linux/bin/activate

# You should see (venv-linux) in your prompt now
```

---

## 📚 Training Commands

### Start Fresh Training
```bash
# Basic training
python src/train.py

# Train for specific number of episodes
python src/train.py --episodes 1000
```

### Resume Training from Checkpoint
```bash
# Resume from a specific checkpoint
python src/train.py --resume checkpoints/dqn_pong_episode_100.h5

# Resume from interrupted training
python src/train.py --resume checkpoints/dqn_pong_interrupted.h5
```

### Run Training in Background (Recommended for Overnight)
```bash
# Run in background - keeps running even if terminal closes

# Save the PID (process ID) shown - you'll need it later
# Example: [1] 12345  <- 12345 is the PID
```

---

## 📊 Monitoring Training

### Check Training Progress
```bash
# View last 20 lines of training output
tail -20 training_*.log

# Watch training live (updates automatically)
tail -f training_*.log

# Press Ctrl+C to stop watching
```

### Actively check training progress
```bash
# This shows episode info + GPU status every 30 seconds.
while true; do 
  echo "========== $(date '+%H:%M:%S') =========="; 
  python -c "import json; d=json.load(open('logs/training_log.json')); print(f'Episode: {d[\"episodes_completed\"]}/3000\nLast reward: {d[\"episode_rewards\"][-1]:.0f}\nAvg (last 10): {sum(d[\"episode_rewards\"][-10:])/10:.1f}')"; 
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; 
  echo ""; 
  sleep 30; 
done
```
# This shows episode info + GPU status every 30 seconds.

### Check Episode Rewards
```bash
# View training log file
cat logs/training_log.json

# See last 10 episode rewards
python -c "import json; data=json.load(open('logs/training_log.json')); print('Last 10 rewards:', data['episode_rewards'][-10:])"

# See current episode number
python -c "import json; data=json.load(open('logs/training_log.json')); print('Current episode:', data['episodes_completed'])"
```

### Check if Agent is Learning (IMPORTANT!)
```bash
# Check if loss is DECREASING over time (key indicator of learning!)
python -c "import json; d=json.load(open('logs/training_log.json')); losses=d['losses']; print(f'First 50 avg loss: {sum(losses[:50])/50:.4f}'); print(f'Last 50 avg loss: {sum(losses[-50:])/50:.4f}')"

# If last loss < first loss: Agent is learning! ✅
# If losses are equal: Agent NOT learning ❌

# Check reward improvement trend
python -c "import json; d=json.load(open('logs/training_log.json')); r=d['episode_rewards']; print(f'First 50 avg: {sum(r[:50])/50:.1f}'); print(f'Last 50 avg: {sum(r[-50:])/50:.1f}')"

# Monitor loss in real-time (updates every 30s)
while true; do
  python -c "import json; d=json.load(open('logs/training_log.json')); ep=d['episodes_completed']; losses=d['losses'][-20:]; avg_loss=sum(losses)/len(losses); rewards=d['episode_rewards'][-10:]; avg_reward=sum(rewards)/10; print(f'Ep {ep} | Loss: {avg_loss:.4f} | Reward (last 10): {avg_reward:.1f}')";
  sleep 30;
done
```

### Check GPU Usage
```bash
# One-time check
nvidia-smi

# Continuous monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Press Ctrl+C to stop
```

### Check System Memory Usage
```bash
# Quick memory check - human readable
free -h

# What to look for:
# - Total: Your WSL2 allocated RAM (should be ~12GB if .wslconfig set)
# - Used: Currently in use by all processes
# - Free: Immediately available
# - Available: Free + reclaimable (the important number!)

# Continuous memory monitoring
watch -n 2 free -h

# Detailed memory breakdown
cat /proc/meminfo | head -20

# Check swap usage
swapon -s
```

### Monitor Training + Memory Together
```bash
# Shows episode progress AND memory in one view
while true; do
  echo "========== $(date '+%H:%M:%S') ==========";
  python -c "import json; d=json.load(open('logs/training_log.json')); print(f'Episode: {d[\"episodes_completed\"]}/3000\nLast reward: {d[\"episode_rewards\"][-1]:.0f}\nAvg (last 10): {sum(d[\"episode_rewards\"][-10:])/10:.1f}')";
  free -h | grep Mem;
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader;
  echo "";
  sleep 30;
done
```

### Check if Training is Running
```bash
# See all Python processes
ps aux | grep python

# See specific training process
ps aux | grep train.py

# Check by PID (if you saved it)
ps -p 12345  # Replace 12345 with your PID
```

### 🎯 ALL-IN-ONE MONITORING (Recommended!)
```bash
# Complete training dashboard - updates every 30 seconds
# Shows: Episode, Loss, Rewards, GPU, Memory - everything you need!
while true; do
  echo "==================== $(date '+%H:%M:%S') ====================";
  python -c "import json; d=json.load(open('logs/training_log.json')); ep=d['episodes_completed']; losses=d['losses'][-20:] if len(d['losses'])>=20 else d['losses']; avg_loss=sum(losses)/len(losses); rewards=d['episode_rewards'][-10:] if len(d['episode_rewards'])>=10 else d['episode_rewards']; avg_reward=sum(rewards)/len(rewards); print(f'Episode: {ep}/3000'); print(f'Loss (last 20): {avg_loss:.4f}'); print(f'Reward (last 10): {avg_reward:.1f}')";
  echo "---";
  free -h | grep "Mem:" | awk '{print "Memory: "$3" used / "$2" total | "$7" available"}';
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F',' '{print "GPU: "$1" util | "$2" / "$3}';
  echo "";
  sleep 30;
done

# Press Ctrl+C to stop monitoring
```

---

## 🎮 Testing Your Agent

### Watch Agent Play (After Episode 100+)
```bash
# Watch agent play 5 games
python src/play.py --checkpoint checkpoints/dqn_pong_episode_100.h5

# Watch 10 games
python src/play.py --checkpoint checkpoints/dqn_pong_episode_100.h5 --episodes 10

# Watch best model
python src/play.py --checkpoint checkpoints/best_model.h5

# Record video of gameplay
python src/play.py --checkpoint checkpoints/best_model.h5 --record
```

### Visualize Training Progress
```bash
# Show interactive training graphs
python utils/visualization.py

# Save graphs to file
python utils/visualization.py --save my_progress.png

# Simple reward-only plot
python utils/visualization.py --simple
```

---

## 🛠️ Managing Background Training

### Stop Training
```bash
# Stop by PID (if you saved it)
kill 12345  # Replace with your PID

# Stop all Python training processes
pkill -f train.py

# Force kill if needed
kill -9 12345  # Replace with your PID
```

### Find Background Training Process
```bash
# List all Python processes with details
ps aux | grep python

# Find training specifically
ps aux | grep train.py
```

---

## 📁 File and Directory Commands

### Check What Files Exist
```bash
# List all checkpoints
ls -lh checkpoints/

# List all log files
ls -lh logs/

# List all videos
ls -lh videos/

# Show file sizes in human-readable format
ls -lh
```

### View File Contents
```bash
# View training log
cat logs/training_log.json

# View last 50 lines of any file
tail -50 filename.log

# View first 20 lines
head -20 filename.log
```

### Clean Up Old Files (If Needed)
```bash
# Remove specific checkpoint
rm checkpoints/dqn_pong_episode_100.h5

# Remove all checkpoints (CAREFUL!)
rm checkpoints/*.h5

# Remove all logs (CAREFUL!)
rm logs/*.json
```

---

## ⚙️ Configuration Changes

### Edit Training Settings
```bash
# Open config file in nano editor
nano src/config.py

# Save: Ctrl+O, then Enter
# Exit: Ctrl+X

# Or open in vim
vim src/config.py

# Save in vim: Press Esc, type :wq, press Enter
```

### Common Settings to Change
```python
# In src/config.py:

MAX_EPISODES = 3000     # Line 106 - Total episodes to train
FRAME_SKIP = 4          # Line 156 - Speed vs quality
BATCH_SIZE = 32         # Line 64 - Training batch size
SAVE_FREQ = 100         # Line 113 - How often to save checkpoints
PRINT_FREQ = 10         # Line 124 - How often to print progress
```

---

## 🔍 Troubleshooting

### Check Python and Package Versions
```bash
# Python version
python --version

# TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# List all installed packages
pip list

# Check specific package
pip show tensorflow
```

### GPU Not Working?
```bash
# Verify GPU detection
python verify_gpu.py

# Check TensorFlow GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Reinstall TensorFlow with GPU
pip install --upgrade tensorflow[and-cuda]
```

### Out of Memory?
```bash
# Check system memory
free -h

# Check disk space
df -h

# Reduce batch size in config.py:
BATCH_SIZE = 16  # Instead of 32
REPLAY_BUFFER_SIZE = 50000  # Instead of 100000
```

### Training Crashed?
```bash
# Check if log file has error at the end
tail -100 training_*.log

# Look for Python errors
grep -i "error\|exception\|traceback" training_*.log

# Check system logs
dmesg | tail -50
```

---

## 💾 Backup and Export

### Backup Checkpoints
```bash
# Copy checkpoints to backup folder
mkdir -p backups
cp checkpoints/*.h5 backups/

# Copy with timestamp
cp checkpoints/best_model.h5 "backups/best_model_$(date +%Y%m%d).h5"
```

### Export Training Data
```bash
# Copy logs to backup
cp logs/training_log.json "backups/training_log_$(date +%Y%m%d).json"

# Create full backup
tar -czf "backup_$(date +%Y%m%d).tar.gz" checkpoints/ logs/ videos/
```

---

## 🎯 Training Workflow (Recommended)

### Day 1: Start Training
```bash
# 1. Navigate and activate
cd "/mnt/c/Users/renbou/Desktop/Most of the Good Stuff/projects/REINFORCEMENT LEARNING PONG"
source venv-linux/bin/activate

# 2. Start background training
nohup python src/train.py > training_logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3. Note the PID shown (e.g., [1] 12345)

# 4. Verify it's running
tail -f training_*.log
# Press Ctrl+C when you see training progress

# 5. Close terminal - training continues!
```

### Later: Check Progress
```bash
# 1. Navigate and activate
cd "/mnt/c/Users/renbou/Desktop/Most of the Good Stuff/projects/REINFORCEMENT LEARNING PONG"
source venv-linux/bin/activate

# 2. Check if training is running
ps aux | grep train.py

# 3. See latest progress
tail -50 training_*.log

# 4. Check GPU usage
nvidia-smi

# 5. View episode count
python -c "import json; print('Current episode:', json.load(open('logs/training_log.json'))['episodes_completed'])"
```

### After Episode 100+: Test Agent
```bash
# 1. Activate venv
cd "/mnt/c/Users/renbou/Desktop/Most of the Good Stuff/projects/REINFORCEMENT LEARNING PONG"
source venv-linux/bin/activate

# 2. Watch agent play
python src/play.py --checkpoint checkpoints/best_model.weights.h5

# 3. Visualize progress
python utils/visualization.py --save progress.png
```

---

## 📝 Quick Tips

### Keyboard Shortcuts
- **Ctrl+C**: Stop current command/process
- **Ctrl+Z**: Pause current process
- **Ctrl+D**: Exit terminal
- **Ctrl+L**: Clear screen
- **Tab**: Auto-complete file/folder names
- **↑/↓ arrows**: Navigate command history

### Useful Aliases (Add to ~/.bashrc for shortcuts)
```bash
# Add these to ~/.bashrc:
alias cdpong='cd "/mnt/c/Users/renbou/Desktop/Most of the Good Stuff/projects/REINFORCEMENT LEARNING PONG"'
alias activate='source venv-linux/bin/activate'
alias checkgpu='nvidia-smi'
alias trainlog='tail -f training_*.log'

# Then reload:
source ~/.bashrc

# Now you can just type: cdpong
```

---

## ⏱️ Expected Timeline

| Episodes | Time Estimate | Agent Behavior |
|----------|---------------|----------------|
| 0-100 | 1-2 hours | Random play, losing badly |
| 100-500 | 3-5 hours | Starting to track ball |
| 500-1000 | 8-12 hours | Competent play, some wins |
| 1000-2000 | 16-24 hours | Strong performance |
| 2000-3000 | 24-36 hours | Mastery, consistent wins |

**Full training**: ~6-8 hours per 1000 episodes on RTX 3050

---

## 🆘 Emergency Commands

### Training Won't Stop
```bash
# Nuclear option - kill all Python
pkill -9 python

# Or reboot WSL2
wsl --shutdown
# Then reopen terminal
```

### Reset Everything
```bash
# Delete all training data (CAREFUL!)
rm -rf checkpoints/* logs/* videos/*

# Start completely fresh
python src/train.py
```

### Windows Going to Sleep?
**Windows Settings:**
1. Settings → System → Power & Battery
2. When plugged in, put device to sleep: **Never**
3. Screen and sleep → When plugged in, turn screen off: **Never**

---

## 📞 Support

If something goes wrong:
1. Check the log files: `tail -100 training_*.log`
2. Check GPU status: `nvidia-smi`
3. Verify process is running: `ps aux | grep train.py`
4. Check this reference guide!

**Last updated**: 2025-12-30
