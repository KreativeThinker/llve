import torch
import numpy as np
from .fourier import amplitude_phase_split, recombine, zero_frequency_component
from .rewards import ProxyQualityScorer, combined_reward


class LowLightEnv:
    def __init__(self, image_initial, scorer=None, max_steps=10, zfc_target=2.5e5, device='cpu', return_image_state=True):
        self.image_initial = torch.from_numpy(image_initial).float() if isinstance(image_initial, np.ndarray) else image_initial.float()
        self.orig_ndim = self.image_initial.ndim
        self.orig_shape = self.image_initial.shape
        self.device = torch.device(device)
        self.scorer = scorer or ProxyQualityScorer()
        self.max_steps = max_steps
        self.zfc_target = zfc_target
        self.return_image_state = return_image_state

        if self.image_initial.ndim == 2:
            self.image_initial = self.image_initial.unsqueeze(0).unsqueeze(0)
        elif self.image_initial.ndim == 3:
            self.image_initial = self.image_initial.unsqueeze(0)

        self.image_initial = self.image_initial.to(self.device)
        self.amp_0, self.pha_0 = amplitude_phase_split(self.image_initial)
        self.amp_0 = self.amp_0.to(self.device)
        self.pha_0 = self.pha_0.to(self.device)
        
        score_initial = self.scorer.score(self.image_initial)
        if isinstance(score_initial, (float, int)):
            self.score_initial = torch.tensor(float(score_initial), device=self.device)
        else:
            self.score_initial = score_initial.to(self.device)
            
        self.reset()

    def reset(self):
        self.image_current = self.image_initial.clone().to(self.device)
        self.amp_current = self.amp_0.clone().to(self.device)
        self.zfc_current = zero_frequency_component(self.amp_current)
        self.step_count = 0
        state = self.image_current if self.return_image_state else self.amp_current
        state = state.clone()
        if self.orig_ndim == 2:
            return state.squeeze(0).squeeze(0)
        elif self.orig_ndim == 3 and not self.return_image_state:
            return state.squeeze(0)
        return state

    def step(self, action_map):
        action_map = action_map.to(self.device).float()
        if action_map.ndim == 2 and self.amp_current.ndim == 4:
            action_map = action_map.unsqueeze(0).unsqueeze(0)
        elif action_map.ndim == 3 and self.amp_current.ndim == 4:
            action_map = action_map.unsqueeze(0)

        assert action_map.shape[-2:] == self.amp_current.shape[-2:], f"action_map shape {action_map.shape} != amp spatial shape {self.amp_current.shape[-2:]}"

        alpha_min, alpha_max, alpha_step = -0.1, 0.2, 0.01
        num_actions = int((alpha_max - alpha_min) / alpha_step) + 1
        assert action_map.max() < num_actions, f"action_map contains {action_map.max()}, max allowed {num_actions - 1}"

        alphas = alpha_min + action_map * alpha_step
        gain = torch.exp(alphas)
        if gain.ndim == 3 and self.amp_current.ndim == 4:
            gain = gain.unsqueeze(1)
        elif gain.ndim == 4 and gain.shape[1] == 1 and self.amp_current.shape[1] > 1:
            gain = gain.expand(-1, self.amp_current.shape[1], -1, -1)

        self.amp_current = (self.amp_current * gain).to(self.device)
        self.image_current = recombine(self.amp_current, self.pha_0)
        self.image_current = torch.clamp(self.image_current, 0, 1)

        self.zfc_current = zero_frequency_component(self.amp_current)
        
        zfc_in = self.zfc_current.mean(dim=1) if self.zfc_current.ndim > 1 else self.zfc_current
        reward = combined_reward(
            self.scorer,
            self.image_current,
            self.image_initial,
            zfc_in,
            self.zfc_target,
            score_initial=self.score_initial
        )
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        out_state = self.image_current if self.return_image_state else self.amp_current
        out_state = out_state.clone()
        if self.orig_ndim == 2:
            ret_reward = reward.item() if reward.numel() == 1 else reward.squeeze().item()
            return out_state.squeeze(0).squeeze(0), ret_reward, done

        return out_state, reward.to(self.device), done

    def get_zfc(self):
        return self.zfc_current
