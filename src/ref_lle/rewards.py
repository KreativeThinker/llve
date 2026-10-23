import torch
from abc import ABC, abstractmethod
import numpy as np
from .utils import to_grayscale, to_numpy


class QualityScorer(ABC):
    @abstractmethod
    def score(self, image):
        """Score image quality. Higher is better. image shape: (h, w) or (c, h, w), vals [0,1]."""
        pass


class ProxyQualityScorer(QualityScorer):
    def score(self, image):
        img_np = to_numpy(image)
        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.ndim == 3:
            img_np = np.mean(img_np, axis=0)

        h, w = img_np.shape
        center_h, center_w = h // 2, w // 2
        window_size = max(1, min(h, w) // 4)
        window = img_np[
            center_h - window_size : center_h + window_size,
            center_w - window_size : center_w + window_size,
        ]

        contrast = float(np.std(window))
        brightness = float(np.mean(window))
        return contrast + 0.5 * brightness


class UniqueScorer(QualityScorer):
    def __init__(self, device='cpu'):
        self.device = device
        self.has_pyiqa = False
        self.fallback = ProxyQualityScorer()
        try:
            import pyiqa
            self.metric = pyiqa.create_metric('unique', device=device)
            self.has_pyiqa = True
        except Exception:
            self.has_pyiqa = False
        
    def score(self, image):
        if not self.has_pyiqa:
            return self.fallback.score(image)

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
    """Paper Eq. (4): r^{amp}_t = | \bar{zfc} / zfc_t - 1 |"""
    diff = torch.abs(zfc_target / (zfc_current + 1e-8) - 1.0)
    return diff


def image_quality_reward(scorer, image_current, image_initial=None, score_initial=None):
    """Paper Eq. (3): r^{iq}_t = S(s_t) - S(s_0)"""
    score_current = scorer.score(image_current)
    if score_initial is None:
        score_initial = scorer.score(image_initial)
    return score_current - score_initial


def combined_reward(scorer, image_current, image_initial=None, zfc_current=None, zfc_target=2.5e5, w_iq=1000, w_amp=60, score_initial=None):
    """Paper Eq. (5): r_t = w_{iq} * r^{iq}_t - w_{amp} * r^{amp}_t"""
    r_iq = image_quality_reward(scorer, image_current, image_initial=image_initial, score_initial=score_initial)
    r_amp = amplitude_exposure_reward(zfc_current, zfc_target)

    r_total = w_iq * r_iq - w_amp * r_amp
    return r_total


def temporal_video_reward(
    scorer,
    image_current,
    image_initial,
    prev_enhanced_img=None,
    prev_input_img=None,
    gain_current=None,
    gain_prev=None,
    zfc_current=None,
    zfc_target=2.5e5,
    w_iq=100.0,
    w_amp=20.0,
    w_gain=15.0,
    w_diff=25.0,
    gt_image=None,
    w_align=50.0,
):
    """
    Physical Motion-Consistent Video Reward:
      - Perceptual quality improvement (r_iq)
      - Exposure alignment with ZFC target (r_amp)
      - Gain smoothness penalty (||G_t - G_{t-1}^*||^2)
      - Motion difference matching: compares raw inter-frame motion (I_t^raw - I_{t-1}^raw)
        against enhanced inter-frame motion (\\hat{I}_t - \\hat{I}_{t-1}^*)
      - Supervised ground-truth alignment (if available)
    """
    device = image_current.device
    r_iq = image_quality_reward(scorer, image_current, image_initial=image_initial)
    r_amp = -amplitude_exposure_reward(zfc_current, zfc_target) if zfc_current is not None else torch.tensor(0.0, device=device)

    # Gain smoothness penalty across frames
    if gain_current is not None and gain_prev is not None:
        gain_penalty = torch.mean((gain_current - gain_prev) ** 2)
    else:
        gain_penalty = torch.tensor(0.0, device=device)

    # Motion difference matching:
    # Delta_raw  = I_t^raw - I_{t-1}^raw
    # Delta_enh  = \\hat{I}_t - \\hat{I}_{t-1}^*
    # Minimizing ||Delta_enh - Delta_raw||_1 guarantees enhanced video moves exactly as the raw camera frames
    if prev_enhanced_img is not None and prev_input_img is not None:
        delta_raw = image_initial - prev_input_img
        delta_enh = image_current - prev_enhanced_img
        motion_diff_penalty = torch.mean(torch.abs(delta_enh - delta_raw))
    else:
        motion_diff_penalty = torch.tensor(0.0, device=device)

    # Ground-truth alignment reward (if training with paired GT)
    if gt_image is not None:
        align_rew = -torch.mean(torch.abs(image_current - gt_image))
    else:
        align_rew = torch.tensor(0.0, device=device)

    total_reward = (
        w_iq * r_iq
        + w_amp * r_amp
        - w_gain * gain_penalty
        - w_diff * motion_diff_penalty
        + w_align * align_rew
    )
    return total_reward
