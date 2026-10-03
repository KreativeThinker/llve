import torch
import numpy as np
from ref_lle.env import LowLightEnv
from ref_lle.rewards import ProxyQualityScorer


def test_env_reset():
    image = torch.rand(64, 64)
    env = LowLightEnv(image, scorer=ProxyQualityScorer(), max_steps=5)
    state = env.reset()

    assert state.shape == image.shape
    assert state.dtype == torch.float32


def test_env_step():
    image = torch.rand(64, 64)
    env = LowLightEnv(image, scorer=ProxyQualityScorer(), max_steps=5)
    state = env.reset()

    action_map = torch.randint(0, 31, state.shape, dtype=torch.int64)
    next_state, reward, done = env.step(action_map)

    assert next_state.shape == state.shape
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert done == False


def test_env_max_steps():
    image = torch.rand(64, 64)
    env = LowLightEnv(image, scorer=ProxyQualityScorer(), max_steps=3)
    state = env.reset()

    done = False
    steps = 0
    while not done and steps < 5:
        action_map = torch.randint(0, 31, state.shape, dtype=torch.int64)
        state, _, done = env.step(action_map)
        steps += 1

    assert done == True
    assert steps == 3


def test_env_amplitude_scaling():
    image = torch.ones(64, 64) * 0.5
    env = LowLightEnv(image, scorer=ProxyQualityScorer(), max_steps=10)
    state = env.reset()

    action_map = torch.ones_like(state, dtype=torch.int64) * 20
    next_state, _, _ = env.step(action_map)

    brightness_original = state.mean()
    brightness_new = next_state.mean()

    assert brightness_new > brightness_original


if __name__ == "__main__":
    test_env_reset()
    print("✓ test_env_reset")

    test_env_step()
    print("✓ test_env_step")

    test_env_max_steps()
    print("✓ test_env_max_steps")

    test_env_amplitude_scaling()
    print("✓ test_env_amplitude_scaling")

    print("\nAll env tests passed!")
