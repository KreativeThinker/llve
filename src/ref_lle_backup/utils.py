import torch
import numpy as np


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x


def ensure_3d(x):
    x = to_numpy(x)
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    return x


def to_grayscale(img):
    img = to_numpy(img)
    if img.ndim == 3:
        return np.mean(img, axis=0)
    return img


def ensure_4d_batched(x):
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x

    if x_np.ndim == 2:
        x_np = np.expand_dims(np.expand_dims(x_np, 0), 0)
    elif x_np.ndim == 3:
        x_np = np.expand_dims(x_np, 1)

    return torch.from_numpy(x_np).float()


def compute_entropy(logits):
    log_softmax = torch.log_softmax(logits, dim=-1)
    softmax = torch.exp(log_softmax)
    entropy = -(softmax * log_softmax + 1e-8).sum(dim=-1).mean()
    return entropy


def sample_action(policy_logits):
    policy = torch.softmax(policy_logits, dim=-1)
    action_dist = torch.distributions.Categorical(policy)
    return action_dist.sample()


def reshape_action(action, state_shape):
    if state_shape.dim() == 2:
        return action.reshape(state_shape)
    else:
        return action.reshape(state_shape[1:])
