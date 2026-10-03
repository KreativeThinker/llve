import torch
import numpy as np
from .fourier import amplitude_phase_split, recombine, zero_frequency_component
from .rewards import combined_reward, ProxyQualityScorer


class LowLightEnv:
    def __init__(self, image_initial, scorer=None, max_steps=10, zfc_target=2.5e5):
        self.image_initial = torch.from_numpy(image_initial).float() if isinstance(image_initial, np.ndarray) else image_initial.float()
        self.scorer = scorer or ProxyQualityScorer()
        self.max_steps = max_steps
        self.zfc_target = zfc_target

        if self.image_initial.ndim == 2:
            self.image_initial = self.image_initial.unsqueeze(0)

        self.amp_0, self.pha_0 = amplitude_phase_split(self.image_initial)
        self.score_initial = self.scorer.score(self.image_initial)
        self.reset()

    def reset(self):
        self.image_current = self.image_initial.clone()
        self.amp_current = self.amp_0.clone()
        self.zfc_current = zero_frequency_component(self.amp_current)
        self.step_count = 0
        return self.image_current.clone()

    def step(self, action_map):
        assert action_map.shape == self.amp_current.shape, f"action_map shape {action_map.shape} != amp shape {self.amp_current.shape}"

        alpha_min, alpha_max, alpha_step = -0.1, 0.2, 0.01
        num_actions = int((alpha_max - alpha_min) / alpha_step) + 1
        assert action_map.max() < num_actions, f"action_map contains {action_map.max()}, max allowed {num_actions - 1}"

        action_map = action_map.float()
        alphas = alpha_min + action_map * alpha_step
        gain = torch.exp(alphas)

        self.amp_current = self.amp_0 * gain
        self.image_current = recombine(self.amp_current, self.pha_0)
        self.image_current = torch.clamp(self.image_current, 0, 1)

        self.zfc_current = zero_frequency_component(self.amp_current)
        reward = combined_reward(
            self.scorer,
            self.image_current,
            self.image_initial,
            self.zfc_current[0],
            self.zfc_target
        )

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.image_current.clone(), reward, done

    def get_zfc(self):
        return self.zfc_current
