import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class RefLLENet(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=32, num_actions=31):
        super().__init__()
        self.encoder = Encoder(input_channels, hidden_dim)
        self.hidden_dim = hidden_dim

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_actions, kernel_size=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        # Format input tensor to [B, 3, H, W]
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        elif x.ndim == 3:
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            x = x.unsqueeze(0)
        elif x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

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


class TemporalDiffRefLLENet(nn.Module):
    """
    Causal Temporal Difference ReF-LLE Policy Network.
    Inputs (9 channels total):
      - amp_cur: Current frame Fourier amplitude A_t (3 channels)
      - amp_prev_updated: Previous frame updated Fourier amplitude A^*_{t-1} (3 channels)
      - delta_prev_updated: Causal difference with previous updated frame I_t - \\hat{I}^*_{t-1} (3 channels)
    Warm-initialized from model_paper_10k.pt.
    """
    def __init__(self, hidden_dim=32, num_actions=31):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.encoder = nn.Module()
        self.encoder.conv1 = nn.Conv2d(9, hidden_dim, kernel_size=3, padding=1)
        self.encoder.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.encoder.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

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

    def load_from_10k(self, ckpt_path, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)
        with torch.no_grad():
            self.encoder.conv1.weight.zero_()
            # Copy 3-channel weights into first 3 channels
            self.encoder.conv1.weight[:, 0:3, :, :] = ckpt["encoder.conv1.weight"]
            # Small random weights for temporal channels (3:9) to smoothly blend temporal delta
            self.encoder.conv1.weight[:, 3:9, :, :] = torch.randn_like(self.encoder.conv1.weight[:, 3:9, :, :]) * 1e-4
            self.encoder.conv1.bias.copy_(ckpt["encoder.conv1.bias"])

            self.encoder.conv2.weight.copy_(ckpt["encoder.conv2.weight"])
            self.encoder.conv2.bias.copy_(ckpt["encoder.conv2.bias"])
            self.encoder.conv3.weight.copy_(ckpt["encoder.conv3.weight"])
            self.encoder.conv3.bias.copy_(ckpt["encoder.conv3.bias"])

            self.policy_head[0].weight.copy_(ckpt["policy_head.0.weight"])
            self.policy_head[0].bias.copy_(ckpt["policy_head.0.bias"])
            self.policy_head[2].weight.copy_(ckpt["policy_head.2.weight"])
            self.policy_head[2].bias.copy_(ckpt["policy_head.2.bias"])

            self.value_head[0].weight.copy_(ckpt["value_head.0.weight"])
            self.value_head[0].bias.copy_(ckpt["value_head.0.bias"])
            self.value_head[2].weight.copy_(ckpt["value_head.2.weight"])
            self.value_head[2].bias.copy_(ckpt["value_head.2.bias"])

    def forward(self, amp_cur, amp_prev_updated=None, delta_prev_updated=None):
        if amp_prev_updated is None and amp_cur.shape[1] == 9:
            x = amp_cur
        else:
            if amp_prev_updated is None:
                amp_prev_updated = amp_cur
            if delta_prev_updated is None:
                delta_prev_updated = torch.zeros_like(amp_cur)

            x = torch.cat([
                torch.log1p(amp_cur),
                torch.log1p(amp_prev_updated),
                delta_prev_updated,
            ], dim=1)

        h = self.relu(self.encoder.conv1(x))
        h = self.relu(self.encoder.conv2(h))
        h = self.relu(self.encoder.conv3(h))

        policy_logits = self.policy_head(h)
        value = self.value_head(h)

        b, c, h_dim, w_dim = policy_logits.shape
        policy_logits = policy_logits.permute(0, 2, 3, 1).reshape(b * h_dim * w_dim, c)
        value = value.permute(0, 2, 3, 1).reshape(b * h_dim * w_dim, 1)

        return policy_logits, value
