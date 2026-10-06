import torch
import torch.nn as nn
from collections import deque
from .utils import compute_entropy


def compute_returns(rewards, values, gamma=0.95):
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().tolist()
    if isinstance(values, torch.Tensor):
        values = values.cpu().tolist()

    ret = deque()
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t < len(values) - 1 else 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * 0.95 * gae
        ret.appendleft(gae + values[t])

    return list(ret)


class A3CWorker:
    def __init__(self, model, optimizer, gamma=0.95, entropy_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def compute_loss(self, states, actions, rewards, log_probs, next_state=None):
        b, h, w = actions.shape
        actions_flat = actions.reshape(-1)
        log_probs_flat = log_probs.reshape(-1)

        policy_logits, values = self.model(states)
        values = values.squeeze(-1)

        device = values.device
        returns = torch.tensor(compute_returns(rewards, values.detach()), dtype=torch.float32, device=device)
        advantages = returns - values.detach()

        policy_loss = -(log_probs_flat * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy = compute_entropy(policy_logits)
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        return loss, policy_loss, value_loss, entropy


def train_step(model, optimizer, state, action_map, reward, gamma=0.95, entropy_coef=0.01):
    optimizer.zero_grad()
    policy_logits, value = model(state)

    action_flat = action_map.reshape(-1) if action_map.ndim > 1 else action_map
    log_softmax = torch.log_softmax(policy_logits, dim=-1)
    log_probs = log_softmax.gather(1, action_flat.unsqueeze(-1)).squeeze(-1)

    value_flat = value.squeeze(-1)
    advantage = reward - value_flat.detach()

    policy_loss = -(log_probs * advantage).mean()
    value_loss = 0.5 * advantage.pow(2).mean()
    entropy = compute_entropy(policy_logits)
    loss = policy_loss + value_loss - entropy_coef * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()
