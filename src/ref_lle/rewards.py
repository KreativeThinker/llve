import torch
from abc import ABC, abstractmethod
import numpy as np
from .utils import to_grayscale


class QualityScorer(ABC):
    @abstractmethod
    def score(self, image):
        """Score image quality. Higher is better. image shape: (h, w) or (c, h, w), vals [0,1]."""
        pass


class ProxyQualityScorer(QualityScorer):
    def score(self, image):
        image = to_grayscale(image)
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        window_size = min(h, w) // 4
        window = image[
            center_h - window_size:center_h + window_size,
            center_w - window_size:center_w + window_size
        ]

        contrast = np.std(window)
        brightness = np.mean(window)
        return float(contrast + 0.5 * brightness)


class UniqueScorer(QualityScorer):
    def __init__(self, device='cpu'):
        import pyiqa
        # Load the pyiqa UNIQUE metric
        self.metric = pyiqa.create_metric('unique', device=device)
        self.device = device
        
    def score(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
            
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        elif image.ndim == 3:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            
        image = image.to(self.device)
        
        with torch.no_grad():
            score_val = self.metric(image)
            
        if score_val.ndim == 0 or score_val.shape[0] == 1:
            return score_val.item()
        return score_val


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

    r_total = w_iq * r_iq - w_amp * r_amp
    return r_total
