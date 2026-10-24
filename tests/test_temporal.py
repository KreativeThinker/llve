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


def test_sdsd_video_dataset_init():
    from ref_lle.video_data import SDSDVideoDataset
    ds = SDSDVideoDataset(root_dir="data/sdsd/indoor_np", clip_len=4)
    assert len(ds) >= 1
    clip_in, clip_gt = ds[0]
    assert clip_in.shape == (4, 3, 128, 128)


def test_resolve_sdsd_input_and_scene(tmp_path):
    from ref_lle.infer_video import resolve_sdsd_input_and_scene
    fake_root = tmp_path / "sdsd_fake"
    in_dir = fake_root / "input" / "test_scene"
    in_dir.mkdir(parents=True)
    (in_dir / "0001.npy").write_bytes(b"dummy")

    resolved_in, resolved_gt, scene = resolve_sdsd_input_and_scene(data_root=str(fake_root), scene="test_scene")
    assert resolved_in == in_dir
    assert scene == "test_scene"

    # Test auto-fallback if requested scene doesn't exist
    resolved_in2, _, scene2 = resolve_sdsd_input_and_scene(data_root=str(fake_root), scene="nonexistent")
    assert resolved_in2 == in_dir
    assert scene2 == "test_scene"
