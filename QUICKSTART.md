# Quick Start Guide - DQN Pong Agent

## 🚀 Getting Started in 5 Minutes

### Step 1: Create Virtual Environment (IMPORTANT!)

Open PowerShell in the project directory and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# You should see (venv) in your prompt now
```

### Step 2: Install Dependencies (2 minutes)

With the virtual environment activated:

```powershell
# Install all dependencies
pip install -r requirements.txt
```

### Step 3: Test the Environment (1 minute)

Verify everything works:

```powershell
# Test the Pong environment
python src/environment.py
```

You should see a Pong game window with random gameplay!

### Step 4: Start Training! (2 minutes to start)

```powershell
# Start training your agent
python src/train.py
```

The agent will start learning! You'll see progress updates every 10 episodes.

---

## 📊 What to Expect

### Training Progress

**Episodes 1-100**: Random exploration
- Agent moves randomly
- Loses badly (-21 reward typical)
- Learning basic controls

**Episodes 100-500**: Improvement phase
- Starts tracking the ball
- Occasional successful hits
- Reward improving toward 0

**Episodes 500-1500**: Competent play
- Consistent ball tracking
- Winning some games
- Positive average reward

**Episodes 1500-3000**: Mastery
- Strong performance
- Beats built-in AI consistently
- Reward around +10 to +15

### Training Time

On your RTX 3050:
- **~4-8 hours** for full training (3000 episodes)
- **~1-2 hours** to see noticeable improvement
- You can stop and resume anytime!

---

## 🎮 Commands Cheat Sheet

### Training

```powershell
# Start fresh training
python src/train.py

# Resume from checkpoint
python src/train.py --resume checkpoints/dqn_pong_episode_1000.h5

# Train for specific number of episodes
python src/train.py --episodes 5000
```

### Watching Your Agent

```powershell
# Watch the best model play
python src/play.py --checkpoint checkpoints/best_model.h5

# Watch and record video
python src/play.py --checkpoint checkpoints/best_model.h5 --record

# Watch 10 games
python src/play.py --checkpoint checkpoints/best_model.h5 --episodes 10
```

### Visualizing Progress

```powershell
# Show training dashboard
python utils/visualization.py

# Save plot to file
python utils/visualization.py --save training_progress.png

# Simple reward plot only
python utils/visualization.py --simple
```

### Testing Individual Components

```powershell
# Test environment
python src/environment.py

# Test replay buffer
python src/replay_buffer.py

# Test DQN model
python src/dqn_model.py

# Test configuration
python src/config.py
```

---

## ⚙️ Customizing Training

Edit `src/config.py` to change hyperparameters:

**To train faster (less quality):**
- Increase `FRAME_SKIP` to 6
- Decrease `REPLAY_BUFFER_SIZE` to 50000
- Increase `EPSILON_DECAY` to 0.99

**To train better (slower):**
- Decrease `LEARNING_RATE` to 0.0001
- Increase `BATCH_SIZE` to 64
- Set `USE_DOUBLE_DQN` to True (already default)

**To use less GPU memory:**
- Decrease `BATCH_SIZE` to 16
- Decrease `REPLAY_BUFFER_SIZE` to 50000

---

## 🐛 Troubleshooting

### "No module named 'gymnasium'"
```powershell
pip install gymnasium[atari] gymnasium[accept-rom-license]
```

### "ROM not found"
```powershell
pip install ale-py
python -m ale_py.roms.utils install
```

### GPU not detected
```powershell
# Check if GPU is visible
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

### Training is very slow
- Check Task Manager: GPU should be at 80-100% usage
- If CPU is maxed instead, GPU isn't being used
- Try: `set TF_FORCE_GPU_ALLOW_GROWTH=true` before running

### Out of memory errors
- Reduce `BATCH_SIZE` in config.py
- Reduce `REPLAY_BUFFER_SIZE` in config.py
- Enable `GPU_MEMORY_GROWTH = True` in config.py (already default)

---

## 📁 Important Files

- **`src/config.py`** - All hyperparameters (edit this to tune training)
- **`src/train.py`** - Main training script
- **`checkpoints/best_model.h5`** - Your best trained model
- **`logs/training_log.json`** - Training metrics
- **`docs/learning_guide.md`** - RL concepts explained

---

## 💡 Tips for Success

1. **Be patient**: First 100 episodes will look random - this is normal!

2. **Monitor progress**: Check the logs every 50-100 episodes

3. **Save checkpoints**: Training saves automatically every 100 episodes

4. **Visualize often**: Run visualization.py to see if training is working

5. **Start small**: Let it train for 500 episodes first, then evaluate

6. **Experiment**: Try different hyperparameters in config.py!

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Test environment
3. ✅ Start training
4. ⏳ Wait for ~500 episodes
5. 📊 Visualize progress
6. 🎮 Watch your agent play!
7. ⚙️ Tune hyperparameters
8. 🔄 Train more!

---

**Ready to start? Run:**
```powershell
pip install -r requirements.txt
python src/train.py
```

**Good luck! 🏓🤖**
