import torch
import numpy as np
from .fourier import amplitude_phase_split, recombine, zero_frequency_component
from .rewards import combined_reward, ProxyQualityScorer


class LowLightEnv:
    def __init__(self, image_initial, scorer=None, max_steps=10, zfc_target=2.5e5, device='cpu'):
        self.image_initial = torch.from_numpy(image_initial).float() if isinstance(image_initial, np.ndarray) else image_initial.float()
        self.device = torch.device(device)
        self.scorer = scorer or ProxyQualityScorer()
        self.max_steps = max_steps
        self.zfc_target = zfc_target

        if self.image_initial.ndim == 2:
            self.image_initial = self.image_initial.unsqueeze(0).unsqueeze(0)
        elif self.image_initial.ndim == 3:
            self.image_initial = self.image_initial.unsqueeze(0)

        self.image_initial = self.image_initial.to(self.device)
        self.amp_0, self.pha_0 = amplitude_phase_split(self.image_initial)
        self.amp_0 = self.amp_0.to(self.device)
        self.pha_0 = self.pha_0.to(self.device)
        
        score_initial = self.scorer.score(self.image_initial)
        if isinstance(score_initial, float):
            self.score_initial = torch.tensor(score_initial, device=self.device)
        else:
            self.score_initial = score_initial.to(self.device)
            
        self.reset()

    def reset(self):
        self.image_current = self.image_initial.clone().to(self.device)
        self.amp_current = self.amp_0.clone().to(self.device)
        self.zfc_current = zero_frequency_component(self.amp_current)
        self.step_count = 0
        return self.amp_current.clone()

    def step(self, action_map):
        action_map = action_map.to(self.device).float()
        assert action_map.shape[-2:] == self.amp_current.shape[-2:], f"action_map shape {action_map.shape} != amp spatial shape {self.amp_current.shape[-2:]}"

        alpha_min, alpha_max, alpha_step = -0.1, 0.2, 0.01
        num_actions = int((alpha_max - alpha_min) / alpha_step) + 1
        assert action_map.max() < num_actions, f"action_map contains {action_map.max()}, max allowed {num_actions - 1}"

        alphas = alpha_min + action_map * alpha_step
        gain = torch.exp(alphas)
        if gain.ndim == 3 and self.amp_current.ndim == 4:
            gain = gain.unsqueeze(1)

        self.amp_current = (self.amp_current * gain).to(self.device)
        self.image_current = recombine(self.amp_current, self.pha_0)
        self.image_current = torch.clamp(self.image_current, 0, 1)

        self.zfc_current = zero_frequency_component(self.amp_current)
        
        # Override image_initial score internally in combined_reward?
        # Actually combined_reward recomputes initial score which is a bit slow!
        # But we'll leave it to not break the API, even if it's slow.
        reward = combined_reward(
            self.scorer,
            self.image_current,
            self.image_initial,
            self.zfc_current.mean(dim=1),
            self.zfc_target
        )
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.amp_current.clone(), reward.to(self.device), done

    def get_zfc(self):
        return self.zfc_current
