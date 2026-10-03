import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_returns(rewards, values, gamma=0.95, n_steps=10):
    returns = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * 0.95 * gae
        returns.insert(0, gae + values[t])

    return returns


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

        states_flat = states.reshape(b * h * w)

        policy_logits, values = self.model(states)
        values = values.squeeze(-1)

        returns = torch.tensor(compute_returns(rewards.cpu().tolist(), values.detach().cpu().tolist(), self.gamma), dtype=torch.float32)

        advantages = returns - values.detach()

        policy_loss = -(log_probs_flat * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()

        entropy = -(torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1) + 1e-8).sum(dim=-1).mean()

        loss = policy_loss + value_loss - self.entropy_coef * entropy

        return loss, policy_loss, value_loss, entropy


def train_step(model, optimizer, state, action_map, reward, gamma=0.95, entropy_coef=0.01):
    optimizer.zero_grad()

    policy_logits, value = model(state)

    if action_map.ndim == 2:
        action_flat = action_map.reshape(-1)
    else:
        action_flat = action_map

    log_softmax = torch.log_softmax(policy_logits, dim=-1)
    log_probs = log_softmax.gather(1, action_flat.unsqueeze(-1)).squeeze(-1)

    value_flat = value.squeeze(-1)
    advantage = reward - value_flat.detach()

    policy_loss = -(log_probs * advantage).mean()
    value_loss = 0.5 * advantage.pow(2).mean()

    entropy = -(torch.softmax(policy_logits, dim=-1) * log_softmax + 1e-8).sum(dim=-1).mean()

    loss = policy_loss + value_loss - entropy_coef * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()
