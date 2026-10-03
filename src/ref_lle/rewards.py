import torch
from abc import ABC, abstractmethod
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


class QualityScorer(ABC):
    @abstractmethod
    def score(self, image):
        """Score image quality. Higher is better. image shape: (h, w) or (c, h, w), vals [0,1]."""
        pass


class ProxyQualityScorer(QualityScorer):
    def __init__(self):
        pass

    def score(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.ndim == 3:
            image = np.mean(image, axis=0)

        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        window_size = min(h, w) // 4
        window = image[
            center_h - window_size:center_h + window_size,
            center_w - window_size:center_w + window_size
        ]

        contrast = np.std(window)
        brightness = np.mean(window)
        score = contrast + 0.5 * brightness

        return float(score)


def amplitude_exposure_reward(zfc_current, zfc_target=2.5e5):
    diff = torch.abs(zfc_target / (zfc_current + 1e-8) - 1.0)
    return -diff


def image_quality_reward(scorer, image_current, image_initial):
    score_current = scorer.score(image_current)
    score_initial = scorer.score(image_initial)
    return score_current - score_initial


def combined_reward(scorer, image_current, image_initial, zfc_current, zfc_target=2.5e5, w_iq=1000, w_amp=60):
    r_iq = image_quality_reward(scorer, image_current, image_initial)
    r_amp = amplitude_exposure_reward(zfc_current, zfc_target)

    if isinstance(r_amp, torch.Tensor):
        r_amp = r_amp.item()

    r_total = w_iq * r_iq - w_amp * r_amp
    return r_total
