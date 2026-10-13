import torch
import numpy as np
import torch.nn.functional as F


def to_tensor(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    else:
        img = img.float()
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        if img.shape[0] in [1, 3]:  # (C, H, W)
            img = img.unsqueeze(0)
        else:  # (H, W, C)
            img = img.permute(2, 0, 1).unsqueeze(0)
    elif img.ndim == 4 and img.shape[-1] in [1, 3]:  # (B, H, W, C)
        img = img.permute(0, 3, 1, 2)
    return torch.clamp(img, 0.0, 1.0)


def psnr(img_true, img_pred):
    """Compute PSNR across RGB channels."""
    t1 = to_tensor(img_true)
    t2 = to_tensor(img_pred)
    mse = F.mse_loss(t1, t2)
    if mse == 0:
        return float('inf')
    return float(10.0 * torch.log10(1.0 / mse).item())


def rgb_to_y(img):
    """Convert RGB to Y channel (BT.601 standard for PSNR_y benchmark)."""
    t = to_tensor(img)
    if t.shape[1] == 1:
        return t
    r = t[:, 0:1, :, :]
    g = t[:, 1:2, :, :]
    b = t[:, 2:3, :, :]
    # Standard ITU-R BT.601 conversion used in image restoration benchmarks
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y


def psnr_y(img_true, img_pred):
    """Compute PSNR on the Y channel (YCbCr) as evaluated in Table I of base paper."""
    y_true = rgb_to_y(img_true)
    y_pred = rgb_to_y(img_pred)
    mse = F.mse_loss(y_true, y_pred)
    if mse == 0:
        return float('inf')
    return float(10.0 * torch.log10(1.0 / mse).item())


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device='cpu'):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g.unsqueeze(1) @ g.unsqueeze(0)
    w = w2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return w


def ssim(img_true, img_pred, window_size=11, sigma=1.5):
    """Compute structural similarity index (SSIM) in PyTorch."""
    t1 = to_tensor(img_true)
    t2 = to_tensor(img_pred)
    device = t1.device
    channels = t1.shape[1]

    window = _gaussian_window(window_size, sigma, channels, device)

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = F.conv2d(t1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(t2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(t1 * t1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(t2 * t2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(t1 * t2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def niqe(img, device='cpu'):
    """Compute NIQE score via pyiqa if available, else standard fallback."""
    try:
        import pyiqa
        metric = pyiqa.create_metric('niqe', device=device)
        t = to_tensor(img).to(device)
        with torch.no_grad():
            score = metric(t)
        return float(score.item())
    except Exception:
        # Heuristic fallback if pyiqa weights/backends not loaded
        t = to_tensor(img)
        gray = 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]
        std = float(torch.std(gray).item())
        mean = float(torch.mean(gray).item())
        return float(1.0 / (std + 1e-5) * (1.0 - mean))


def lpips_score(img_true, img_pred, device='cpu'):
    """Compute LPIPS score via pyiqa or fallback."""
    try:
        import pyiqa
        metric = pyiqa.create_metric('lpips', device=device)
        t1 = to_tensor(img_true).to(device)
        t2 = to_tensor(img_pred).to(device)
        with torch.no_grad():
            score = metric(t1, t2)
        return float(score.item())
    except Exception:
        return 0.0


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def add(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def average(self, name):
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0
        return float(np.mean(self.metrics[name]))

    def reset(self):
        self.metrics = {}

    def summary(self):
        return {name: float(np.mean(vals)) for name, vals in self.metrics.items()}
