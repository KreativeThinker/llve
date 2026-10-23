import torch
from .fourier import amplitude_phase_split, recombine, zero_frequency_component
from .rewards import ProxyQualityScorer, combined_reward


class LowLightEnv:
    def __init__(self, image, scorer=None, max_steps=10, zfc_target=2.5e5, device='cpu'):
        self.image_initial = image
        self.orig_ndim = image.ndim
        self.orig_shape = image.shape
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
        state = self.amp_current.clone()
        if self.orig_ndim == 2:
            return state.squeeze(0).squeeze(0)
        elif self.orig_ndim == 3:
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
        
        reward = combined_reward(
            self.scorer,
            self.image_current,
            self.image_initial,
            self.zfc_current.mean(dim=1) if self.zfc_current.ndim > 1 else self.zfc_current,
            self.zfc_target
        )
        if isinstance(reward, torch.Tensor):
            reward = float(reward.mean().item())
        else:
            reward = float(reward)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        state = self.amp_current.clone()
        if self.orig_ndim == 2:
            state = state.squeeze(0).squeeze(0)
        elif self.orig_ndim == 3:
            state = state.squeeze(0)

        return state, reward, done

    def get_zfc(self):
        return self.zfc_current
