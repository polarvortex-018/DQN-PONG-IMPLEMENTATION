# Reinforcement Learning Pong Agent 🏓🤖

A Deep Q-Network (DQN) agent that learns to play Atari Pong through reinforcement learning.

## Project Overview

This project implements a DQN agent from scratch using TensorFlow to master the classic Atari game Pong. The agent learns purely from pixel inputs and game rewards, without any hardcoded game knowledge.

### What You'll Learn

- Reinforcement Learning fundamentals (MDPs, Q-Learning)
- Deep Q-Networks (DQN) algorithm
- Experience Replay and Target Networks
- Training neural networks for game AI
- Hyperparameter tuning for RL

## Project Structure

```
reinforcement-learning-pong/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docs/
│   └── learning_guide.md    # RL concepts explained
├── src/
│   ├── config.py            # Hyperparameters and settings
│   ├── environment.py       # Pong environment wrapper
│   ├── replay_buffer.py     # Experience replay memory
│   ├── dqn_model.py         # Neural network architecture
│   ├── agent.py             # DQN agent implementation
│   └── train.py             # Training loop
├── utils/
│   ├── visualization.py     # Plot training progress
│   └── preprocessing.py     # Frame preprocessing utilities
├── notebooks/
│   └── explore_pong.ipynb   # Interactive environment exploration
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs and metrics
└── videos/                  # Recorded gameplay videos
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install required packages
pip install -r requirements.txt
```

### 2. Verify GPU Setup (Optional but Recommended)

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

Your RTX 3050 should be detected automatically!

### 3. Test Environment

```bash
python src/environment.py
```

This will open a window showing random Pong gameplay to verify everything works.

## Usage

### Training the Agent

```bash
# Start training with default hyperparameters
python src/train.py

# Resume training from checkpoint
python src/train.py --resume --checkpoint checkpoints/dqn_pong_episode_1000.h5

# Train with custom hyperparameters
python src/train.py --episodes 5000 --learning-rate 0.0001
```

### Watching the Agent Play

```bash
# Watch trained agent play
python src/play.py --checkpoint checkpoints/best_model.h5

# Record video
python src/play.py --checkpoint checkpoints/best_model.h5 --record
```

### Visualizing Training Progress

```bash
# Plot training metrics
python utils/visualization.py --log-dir logs/
```

## Training Timeline

Based on typical DQN training on Pong:

- **Episodes 0-100**: Random exploration, learning basic controls
- **Episodes 100-500**: Agent starts tracking ball, occasional hits
- **Episodes 500-1500**: Competent play, winning some games
- **Episodes 1500-3000**: Strong performance, consistent wins
- **Episodes 3000+**: Near-optimal play

**Expected training time**: 4-8 hours on RTX 3050 (depends on settings)

## Hyperparameters

Key parameters you can tune (in `src/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.00025 | How fast the network learns |
| `GAMMA` | 0.99 | Discount factor for future rewards |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.1 | Final exploration rate |
| `EPSILON_DECAY` | 0.995 | Exploration decay per episode |
| `BATCH_SIZE` | 32 | Training batch size |
| `REPLAY_BUFFER_SIZE` | 100000 | Max experiences to store |
| `TARGET_UPDATE_FREQ` | 10000 | Steps between target network updates |

## Troubleshooting

### Common Issues

**1. "No module named 'gymnasium'"**
```bash
pip install gymnasium[atari] gymnasium[accept-rom-license]
```

**2. "ROM not found"**
```bash
pip install gymnasium[accept-rom-license]
```

**3. GPU not detected**
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

**4. Training is slow**
- Verify GPU is being used
- Reduce frame skip or batch size
- Check CPU/GPU usage in Task Manager

## Project Roadmap

- [x] Set up project structure
- [x] Create learning guide
- [ ] Implement environment wrapper
- [ ] Build replay buffer
- [ ] Create DQN model
- [ ] Implement DQN agent
- [ ] Build training loop
- [ ] Add visualization tools
- [ ] Train initial model
- [ ] Optimize hyperparameters
- [ ] Create demo videos

## Resources

- [Learning Guide](docs/learning_guide.md) - RL concepts explained
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Original research
- [Gymnasium Docs](https://gymnasium.farama.org/) - Environment documentation

## License

MIT License - Feel free to use this for learning!

## Acknowledgments

- DeepMind for the DQN algorithm
- OpenAI Gym/Gymnasium for the Pong environment
- The RL community for excellent learning resources

---

**Happy Learning! 🚀**

If you have questions or run into issues, check the learning guide or troubleshooting section above.
