import torch
import numpy as np

from .fourier import amplitude_phase_split, recombine, zero_frequency_component
from .rewards import ProxyQualityScorer, temporal_video_reward


class TemporalVideoLowLightEnv:
    """
    Causal Video Environment for Temporal Difference ReF-LLE.
    Iterates over a sequence of video frames, maintaining:
      - Previous updated frame (\\hat{I}^*_{t-1})
      - Previous updated Fourier amplitude (A^*_{t-1})
      - Previous cumulative gain map (G^*_{t-1})
      - Causal difference with previous updated frame: Delta I_{t, prev_updated} = I_t - \\hat{I}^*_{t-1}
    No future/upcoming frames are used.
    """
    def __init__(
        self,
        video_clip,
        gt_clip=None,
        scorer=None,
        steps_per_frame=3,
        zfc_target=5.8e4,
        alpha_min=-0.1,
        alpha_max=0.35,
        alpha_step=0.015,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.video_clip = video_clip.to(self.device).float()  # (T, 3, H, W)
        self.gt_clip = gt_clip.to(self.device).float() if gt_clip is not None else None
        self.scorer = scorer or ProxyQualityScorer()
        self.steps_per_frame = steps_per_frame
        self.zfc_target = zfc_target

        self.num_frames, self.channels, self.h, self.w = self.video_clip.shape

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.num_actions = int(round((self.alpha_max - self.alpha_min) / self.alpha_step)) + 1

        self.reset_video()

    def reset_video(self):
        self.current_frame_idx = 0
        self.current_step = 0

        self.prev_enhanced_img = None
        self.prev_enhanced_amp = None
        self.prev_gain = None

        return self._prepare_frame(0)

    def _prepare_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        self.current_step = 0

        self.image_initial = self.video_clip[frame_idx : frame_idx + 1]  # (1, 3, H, W)
        self.image_current = self.image_initial.clone()

        self.amp_0, self.pha_0 = amplitude_phase_split(self.image_initial)
        self.amp_current = self.amp_0.clone()
        self.cum_gain = torch.ones_like(self.amp_current)

        if self.prev_enhanced_img is None:
            # First frame: reference is the frame itself
            self.prev_enhanced_img = self.image_initial.clone()
            self.prev_enhanced_amp = self.amp_0.clone()
            self.prev_gain = torch.ones_like(self.amp_current)
            self.delta_prev_updated = torch.zeros_like(self.image_initial)
        else:
            # Causal frame differencing: current raw frame minus previous updated frame
            self.delta_prev_updated = self.image_initial - self.prev_enhanced_img

        return self._get_composite_state()

    def _get_composite_state(self):
        # 9-channel causal composite state: [amp_cur, prev_enhanced_amp, delta_prev_updated]
        return torch.cat([
            torch.log1p(self.amp_current),
            torch.log1p(self.prev_enhanced_amp),
            self.delta_prev_updated,
        ], dim=1)

    def step(self, action_map):
        action_map = action_map.to(self.device).float()
        alphas = self.alpha_min + action_map * self.alpha_step
        step_gain = torch.exp(alphas).unsqueeze(1)  # (1, 1, H, W)

        self.amp_current = self.amp_current * step_gain
        self.cum_gain = self.cum_gain * step_gain

        self.image_current = recombine(self.amp_current, self.pha_0)
        self.image_current = torch.clamp(self.image_current, 0.0, 1.0)

        zfc_cur = zero_frequency_component(self.amp_current).mean()

        gt_frame = self.gt_clip[self.current_frame_idx : self.current_frame_idx + 1] if self.gt_clip is not None else None

        reward = temporal_video_reward(
            scorer=self.scorer,
            image_current=self.image_current,
            image_initial=self.image_initial,
            prev_enhanced_img=self.prev_enhanced_img,
            prev_input_img=self.video_clip[max(0, self.current_frame_idx - 1) : max(0, self.current_frame_idx - 1) + 1],
            gain_current=self.cum_gain,
            gain_prev=self.prev_gain,
            zfc_current=zfc_cur,
            zfc_target=self.zfc_target,
            gt_image=gt_frame,
            w_align=100.0,
        )

        self.current_step += 1
        frame_done = self.current_step >= self.steps_per_frame
        video_done = False

        if frame_done:
            # Update history references with the final enhanced frame
            self.prev_enhanced_img = self.image_current.detach().clone()
            self.prev_enhanced_amp = self.amp_current.detach().clone()
            self.prev_gain = self.cum_gain.detach().clone()

            if self.current_frame_idx + 1 < self.num_frames:
                next_state = self._prepare_frame(self.current_frame_idx + 1)
            else:
                video_done = True
                next_state = self._get_composite_state()
        else:
            # Update delta with previous updated frame
            self.delta_prev_updated = self.image_current - self.prev_enhanced_img
            next_state = self._get_composite_state()

        info = {
            "frame_idx": self.current_frame_idx,
            "step": self.current_step,
            "zfc": zfc_cur.item(),
        }
        return next_state, reward, frame_done, video_done, info
