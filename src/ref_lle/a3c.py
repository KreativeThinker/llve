import torch
import torch.nn as nn
from .utils import compute_entropy


def compute_discounted_returns(rewards, values=None, gamma=0.95):
    """
    Compute discounted cumulative returns:
    R_t = sum_{k=0}^{T - t - 1} gamma^k * r_{t+k} + (gamma^{T-t} * V(s_T) if values)
    """
    T = len(rewards)
    returns = [0.0] * T
    R = 0.0
    if values is not None and len(values) > T:
        R = values[-1].detach()

    for t in reversed(range(T)):
        r = rewards[t]
        R = r + gamma * R
        returns[t] = R

    return returns


def train_trajectory_step(model, optimizer, states, actions, rewards, next_state=None, gamma=0.95, entropy_coef=0.01):
    """
    Train Actor-Critic network across a full episode trajectory.
    states: list of [B, C, H, W] tensors
    actions: list of [B, H, W] integer tensors
    rewards: list of [B] or [B, 1] reward tensors
    """
    optimizer.zero_grad()
    T = len(states)
    assert T > 0

    all_policy_loss = []
    all_value_loss = []
    all_entropy = []

    # Forward pass to get values and logits
    all_values = []
    all_logits = []
    all_log_probs = []

    for t in range(T):
        st = states[t]
        act = actions[t]
        b, h, w = act.shape
        action_flat = act.reshape(-1)

        policy_logits, val = model(st)
        val_flat = val.squeeze(-1)  # (b * h * w,)

        log_softmax = torch.log_softmax(policy_logits, dim=-1)
        log_prob = log_softmax.gather(1, action_flat.unsqueeze(-1)).squeeze(-1)

        all_values.append(val_flat)
        all_logits.append(policy_logits)
        all_log_probs.append(log_prob)

    # Compute value for next_state if terminal state has bootstrap value
    v_last = None
    if next_state is not None:
        with torch.no_grad():
            _, next_val = model(next_state)
            v_last = next_val.squeeze(-1).mean()

    # Compute returns backwards
    R = v_last if v_last is not None else 0.0
    for t in reversed(range(T)):
        r_step = rewards[t]
        if isinstance(r_step, torch.Tensor):
            r_val = r_step.mean().item()
        else:
            r_val = float(r_step)

        R = r_val + gamma * R
        val_t = all_values[t]
        adv = R - val_t.detach()

        # Policy loss & Value loss
        p_loss = -(all_log_probs[t] * adv).mean()
        v_loss = 0.5 * (R - val_t).pow(2).mean()
        ent = compute_entropy(all_logits[t])

        all_policy_loss.append(p_loss)
        all_value_loss.append(v_loss)
        all_entropy.append(ent)

    total_policy_loss = torch.stack(all_policy_loss).mean()
    total_value_loss = torch.stack(all_value_loss).mean()
    total_entropy = torch.stack(all_entropy).mean()

    total_loss = total_policy_loss + 0.5 * total_value_loss - entropy_coef * total_entropy

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
    optimizer.step()

    return total_loss.item(), total_policy_loss.item(), total_value_loss.item(), total_entropy.item()


def train_step(model, optimizer, state, action_map, reward, gamma=0.95, entropy_coef=0.01):
    """Single step update for backward compatibility."""
    return train_trajectory_step(
        model, optimizer,
        states=[state],
        actions=[action_map],
        rewards=[reward],
        gamma=gamma,
        entropy_coef=entropy_coef
    )
