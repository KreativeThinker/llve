import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(img_true, img_pred):
    if isinstance(img_true, torch.Tensor):
        img_true = img_true.cpu().numpy()
    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.cpu().numpy()

    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)

    return peak_signal_noise_ratio(img_true, img_pred, data_range=1.0)


def ssim(img_true, img_pred):
    if isinstance(img_true, torch.Tensor):
        img_true = img_true.cpu().numpy()
    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.cpu().numpy()

    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)

    if img_true.ndim == 3:
        return structural_similarity(img_true, img_pred, channel_axis=0, data_range=1.0)
    else:
        return structural_similarity(img_true, img_pred, data_range=1.0)


def niqe(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    img = np.clip(img, 0, 1)

    if img.ndim == 3:
        img = np.mean(img, axis=0)

    mu = np.mean(img)
    sigma = np.std(img)

    if sigma < 0.01:
        return 10.0

    return mu + 0.5 * sigma


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
        return np.mean(self.metrics[name])

    def reset(self):
        self.metrics = {}

    def summary(self):
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = np.mean(values)
        return summary
