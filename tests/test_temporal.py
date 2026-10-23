import torch
from ref_lle.model import TemporalDiffRefLLENet
from ref_lle.video_env import TemporalVideoLowLightEnv
from ref_lle.rewards import temporal_video_reward, ProxyQualityScorer


def test_temporal_diff_model_forward():
    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31)
    amp_cur = torch.rand(1, 3, 64, 64)
    amp_prev = torch.rand(1, 3, 64, 64)
    delta_raw = torch.rand(1, 3, 64, 64)

    policy_logits, value = model(amp_cur, amp_prev, delta_raw)
    assert policy_logits.shape == (64 * 64, 31)
    assert value.shape == (64 * 64, 1)


def test_temporal_diff_model_10k_loader():
    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31)
    model.load_from_10k("model_paper_10k.pt")
    # Verify conv1 weight shape is (32, 9, 3, 3)
    assert model.encoder.conv1.weight.shape == (32, 9, 3, 3)


def test_temporal_video_env():
    video_clip = torch.rand(4, 3, 64, 64)
    env = TemporalVideoLowLightEnv(video_clip, steps_per_frame=4, warmup_steps=6)
    init_state = env.reset_video()

    assert init_state.shape == (1, 9, 64, 64)

    action_map = torch.randint(0, 31, (1, 64, 64))
    next_state, reward, frame_done, video_done, info = env.step(action_map)

    assert next_state.shape == (1, 9, 64, 64)
    assert isinstance(reward, torch.Tensor)
    assert frame_done is False
    assert video_done is False


def test_temporal_video_reward():
    img_cur = torch.rand(1, 3, 64, 64)
    img_init = torch.rand(1, 3, 64, 64)
    scorer = ProxyQualityScorer()

    reward = temporal_video_reward(
        scorer=scorer,
        image_current=img_cur,
        image_initial=img_init,
        prev_enhanced_img=img_init,
        prev_input_img=img_init,
        gain_current=torch.ones_like(img_cur),
        gain_prev=torch.ones_like(img_cur),
        zfc_current=torch.tensor(1.0),
    )
    assert isinstance(reward, torch.Tensor)
