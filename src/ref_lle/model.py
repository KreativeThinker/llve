import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class RefLLENet(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=32, num_actions=31):
        super().__init__()
        self.encoder = Encoder(input_channels, hidden_dim)
        self.hidden_dim = hidden_dim

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_actions, kernel_size=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        enc = self.encoder(x)

        policy_logits = self.policy_head(enc)
        value = self.value_head(enc)

        b, c, h, w = policy_logits.shape
        policy_logits = policy_logits.permute(0, 2, 3, 1).reshape(b * h * w, c)
        value = value.permute(0, 2, 3, 1).reshape(b * h * w, 1)

        return policy_logits, value

    def get_action(self, x):
        policy_logits, _ = self.forward(x)
        policy = torch.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        return action, policy, action_dist.log_prob(action)
