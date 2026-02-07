import torch
import torchvision.transforms as T
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env, last_obss=None):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float()
    elif env in ['ALE/Pong-v5']:
        # Normalize frames to [0, 1] range
        normalized_frames = np.array([frame / 255.0 for frame in obs])
        
        return torch.tensor(normalized_frames, device=device).float()
    else:
        raise ValueError(
            'Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
