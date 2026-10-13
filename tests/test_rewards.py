import torch
from ref_lle.rewards import amplitude_exposure_reward, combined_reward, ProxyQualityScorer


def test_amplitude_exposure_reward():
    zfc_target = 2.5e5
    zfc_exact = torch.tensor(2.5e5)
    r_exact = amplitude_exposure_reward(zfc_exact, zfc_target)
    assert abs(r_exact.item()) < 1e-4

    zfc_far = torch.tensor(1.25e5)
    r_far = amplitude_exposure_reward(zfc_far, zfc_target)
    assert r_far.item() > 0.5  # Positive penalty


def test_combined_reward_penalty():
    scorer = ProxyQualityScorer()
    img_init = torch.rand(1, 3, 64, 64)
    img_curr = img_init.clone()

    zfc_exact = torch.tensor(2.5e5)
    zfc_far = torch.tensor(1.0e5)

    r_good = combined_reward(scorer, img_curr, img_init, zfc_exact, zfc_target=2.5e5)
    r_bad = combined_reward(scorer, img_curr, img_init, zfc_far, zfc_target=2.5e5)

    # Exposure matching target must yield higher reward than exposure far from target
    assert r_good > r_bad
