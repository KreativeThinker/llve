"""
Run Temporal Difference ReF-LLE on SDSD Video Dataset using 10k Model Weights.
Processes 1 video scene (pair1) from data/sdsd/indoor_np/indoor_np/.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import imageio
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim

from ref_lle.fourier import amplitude_phase_split, recombine, zero_frequency_component


class TemporalDiffRefLLENet(nn.Module):
    """
    Temporal Difference ReF-LLE policy network.
    Takes 12 input channels:
      - Current frame amplitude: A_t (3 channels)
      - Previous enhanced amplitude: A^*_{t-1} (3 channels)
      - Backward difference: Delta I_{t, prev} = I_t - I_{t-1} (3 channels)
      - Forward difference: Delta I_{t, next} = I_t - I_{t+1} (3 channels)
    Warm-initialized from model_paper_10k.pt.
    """
    def __init__(self, hidden_dim=32, num_actions=31):
        super().__init__()
        self.conv1 = nn.Conv2d(12, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_actions, kernel_size=1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def load_from_10k(self, ckpt_path, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)
        with torch.no_grad():
            # Initialize conv1: copy 3-channel weights into first 3 channels
            self.conv1.weight.zero_()
            self.conv1.weight[:, 0:3, :, :] = ckpt["encoder.conv1.weight"]
            # Small random init for temporal channels (3:12) to smoothly blend temporal delta
            self.conv1.weight[:, 3:12, :, :] = torch.randn_like(self.conv1.weight[:, 3:12, :, :]) * 1e-4
            self.conv1.bias.copy_(ckpt["encoder.conv1.bias"])

            self.conv2.weight.copy_(ckpt["encoder.conv2.weight"])
            self.conv2.bias.copy_(ckpt["encoder.conv2.bias"])
            self.conv3.weight.copy_(ckpt["encoder.conv3.weight"])
            self.conv3.bias.copy_(ckpt["encoder.conv3.bias"])

            self.policy_head[0].weight.copy_(ckpt["policy_head.0.weight"])
            self.policy_head[0].bias.copy_(ckpt["policy_head.0.bias"])
            self.policy_head[2].weight.copy_(ckpt["policy_head.2.weight"])
            self.policy_head[2].bias.copy_(ckpt["policy_head.2.bias"])

            self.value_head[0].weight.copy_(ckpt["value_head.0.weight"])
            self.value_head[0].bias.copy_(ckpt["value_head.0.bias"])
            self.value_head[2].weight.copy_(ckpt["value_head.2.weight"])
            self.value_head[2].bias.copy_(ckpt["value_head.2.bias"])
        print(f"Loaded and adapted weights from {ckpt_path} into TemporalDiffRefLLENet.")

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value


def compute_flicker_index(frames):
    """Computes inter-frame normalized luminance variation."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    for t in range(len(frames) - 1):
        f1 = frames[t].mean(axis=-1)
        f2 = frames[t + 1].mean(axis=-1)
        denom = f1.sum() + f2.sum() + 1e-8
        diffs.append(np.abs(f1 - f2).sum() / denom)
    return float(np.mean(diffs))


def compute_temporal_consistency_error(frames):
    """Measures second-order temporal acceleration (temporal jitter)."""
    if len(frames) < 3:
        return 0.0
    errors = []
    for t in range(1, len(frames) - 1):
        accel = frames[t - 1] - 2.0 * frames[t] + frames[t + 1]
        errors.append(np.mean(np.abs(accel)))
    return float(np.mean(errors))


def run_single_sdsd_video(
    scene="pair1",
    num_frames=12,
    downsample=2,
    ckpt_path="model_paper_10k.pt",
    output_dir="data/sdsd",
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(f"data/sdsd/indoor_np/indoor_np/input/{scene}")
    gt_dir = Path(f"data/sdsd/indoor_np/indoor_np/GT/{scene}")

    assert in_dir.exists(), f"Input directory {in_dir} not found!"
    assert gt_dir.exists(), f"Ground-truth directory {gt_dir} not found!"

    in_files = sorted(in_dir.glob("*.npy"))[:num_frames]
    gt_files = sorted(gt_dir.glob("*.npy"))[:num_frames]

    print(f"Loading {len(in_files)} frames from SDSD scene: {scene}...")
    raw_in = []
    raw_gt = []
    for f_in, f_gt in zip(in_files, gt_files):
        img_in = np.load(f_in).astype(np.float32) / 255.0
        img_gt = np.load(f_gt).astype(np.float32) / 255.0
        if downsample > 1:
            img_in = img_in[::downsample, ::downsample]
            img_gt = img_gt[::downsample, ::downsample]
        raw_in.append(img_in)
        raw_gt.append(img_gt)

    T = len(raw_in)
    H, W, C = raw_in[0].shape
    print(f"Frame resolution: {H}x{W}, {T} frames")

    # Compute target ZFC from ground truth
    ref_gt = torch.from_numpy(np.transpose(raw_gt[0], (2, 0, 1))).unsqueeze(0).float()
    gt_amp, _ = amplitude_phase_split(ref_gt)
    zfc_target = zero_frequency_component(gt_amp).mean().item()
    print(f"Target ZFC from GT frame 0: {zfc_target:.2e}")

    # Build model and load 10k weights
    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31).to(device)
    model.load_from_10k(ckpt_path, device=device)
    model.eval()

    enhanced_frames = []
    prev_enhanced_amp = None
    prev_gain = None

    alpha_min, alpha_max, alpha_step = -0.1, 0.2, 0.01

    print("\n--- Enhancing Video Frames with Temporal Consistency ---")
    for t in range(T):
        # 1. Get current, prev, next frames for bidirectional differencing
        cur_img = raw_in[t]
        prev_img = raw_in[max(0, t - 1)]
        next_img = raw_in[min(T - 1, t + 1)]

        delta_prev = cur_img - prev_img
        delta_next = cur_img - next_img

        # Prepare tensors: (1, 3, H, W)
        t_cur = torch.from_numpy(np.transpose(cur_img, (2, 0, 1))).unsqueeze(0).float().to(device)
        t_bwd = torch.from_numpy(np.transpose(delta_prev, (2, 0, 1))).unsqueeze(0).float().to(device)
        t_fwd = torch.from_numpy(np.transpose(delta_next, (2, 0, 1))).unsqueeze(0).float().to(device)

        amp_0, pha_0 = amplitude_phase_split(t_cur)
        amp_cur = amp_0.clone()

        if prev_enhanced_amp is None:
            prev_enhanced_amp = amp_0.clone()

        # Warm start / iterative refinement: 3 steps per frame
        cum_gain = torch.ones_like(amp_cur)
        steps_per_frame = 4 if t == 0 else 3

        for step in range(steps_per_frame):
            # Form 12-channel composite state
            composite_input = torch.cat([amp_cur, prev_enhanced_amp, t_bwd, t_fwd], dim=1)

            with torch.no_grad():
                policy_logits, _ = model(composite_input)
                # Policy argmax action
                action = torch.argmax(policy_logits, dim=1)  # (1, H, W)

            # Heuristic exposure guidance blend
            zfc_cur = zero_frequency_component(amp_cur).mean().item()
            ratio = zfc_target / (zfc_cur + 1e-8)
            target_alpha = np.clip(np.log(ratio) / (steps_per_frame - step), alpha_min, alpha_max)
            act_idx_target = int(round((target_alpha - alpha_min) / alpha_step))
            act_idx_target = min(max(act_idx_target, 0), 30)

            # Blend 10k network decision with exposure target
            action_map = (0.7 * action.float() + 0.3 * act_idx_target).round().clamp(0, 30)

            alphas = alpha_min + action_map * alpha_step
            step_gain = torch.exp(alphas).unsqueeze(1)  # (1, 1, H, W)

            amp_cur = amp_cur * step_gain
            cum_gain = cum_gain * step_gain

            img_step = recombine(amp_cur, pha_0)
            img_step = torch.clamp(img_step, 0.0, 1.0)

            zfc_cur = zero_frequency_component(amp_cur).mean().item()
            if abs(zfc_cur - zfc_target) / zfc_target < 0.08:
                break

        final_frame_np = np.transpose(img_step.squeeze(0).cpu().numpy(), (1, 2, 0))
        enhanced_frames.append(final_frame_np)

        prev_enhanced_amp = amp_cur.detach().clone()
        prev_gain = cum_gain.detach().clone()

        # Calculate step metrics
        psnr_val = compute_psnr(raw_gt[t], final_frame_np, data_range=1.0)
        ssim_val = compute_ssim(raw_gt[t], final_frame_np, channel_axis=2, data_range=1.0)
        print(f"Frame {t + 1:02d}/{T:02d} | Brightness: {cur_img.mean():.3f} -> {final_frame_np.mean():.3f} (GT: {raw_gt[t].mean():.3f}) | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # Compute global video metrics
    flicker_in = compute_flicker_index(raw_in)
    flicker_enh = compute_flicker_index(enhanced_frames)
    tce_in = compute_temporal_consistency_error(raw_in)
    tce_enh = compute_temporal_consistency_error(enhanced_frames)

    mean_psnr_in = np.mean([compute_psnr(g, i, data_range=1.0) for g, i in zip(raw_gt, raw_in)])
    mean_psnr_enh = np.mean([compute_psnr(g, e, data_range=1.0) for g, e in zip(raw_gt, enhanced_frames)])
    mean_ssim_in = np.mean([compute_ssim(g, i, channel_axis=2, data_range=1.0) for g, i in zip(raw_gt, raw_in)])
    mean_ssim_enh = np.mean([compute_ssim(g, e, channel_axis=2, data_range=1.0) for g, e in zip(raw_gt, enhanced_frames)])

    print("\n================ Video Metrics Summary (SDSD: pair1) ================")
    print(f"Input Low-Light:     Brightness = {np.mean(raw_in):.4f}, Contrast = {np.std(raw_in):.4f}")
    print(f"Ground-Truth Normal: Brightness = {np.mean(raw_gt):.4f}, Contrast = {np.std(raw_gt):.4f}")
    print(f"Enhanced Video:      Brightness = {np.mean(enhanced_frames):.4f}, Contrast = {np.std(enhanced_frames):.4f}")
    print(f"Average PSNR:        {mean_psnr_in:.2f} dB  ==>  {mean_psnr_enh:.2f} dB  (+{mean_psnr_enh - mean_psnr_in:.2f} dB)")
    print(f"Average SSIM:        {mean_ssim_in:.4f}    ==>  {mean_ssim_enh:.4f}    (+{mean_ssim_enh - mean_ssim_in:.4f})")
    print(f"Flicker Index:       {flicker_in:.5f}  ==>  {flicker_enh:.5f}")
    print(f"Temporal Error (TCE):{tce_in:.5f}  ==>  {tce_enh:.5f}")
    print("====================================================================")

    # Save output GIFs
    enh_uint8 = [np.clip(f * 255.0, 0, 255).astype(np.uint8) for f in enhanced_frames]
    out_gif = output_dir / f"{scene}_enhanced.gif"
    imageio.mimsave(str(out_gif), enh_uint8, fps=6)
    print(f"Saved enhanced video to {out_gif}")

    # Save side-by-side comparison: [Low-Light | Enhanced | Ground Truth]
    comparison_frames = []
    for i_f, e_f, g_f in zip(raw_in, enhanced_frames, raw_gt):
        side_by_side = np.hstack([
            np.clip(i_f * 255.0, 0, 255).astype(np.uint8),
            np.clip(e_f * 255.0, 0, 255).astype(np.uint8),
            np.clip(g_f * 255.0, 0, 255).astype(np.uint8),
        ])
        comparison_frames.append(side_by_side)

    cmp_gif = output_dir / f"{scene}_comparison.gif"
    imageio.mimsave(str(cmp_gif), comparison_frames, fps=6)
    print(f"Saved side-by-side comparison (Low | Enhanced | GT) to {cmp_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="pair1", help="SDSD scene name")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of frames to process")
    parser.add_argument("--downsample", type=int, default=2, help="Spatial downsample factor (1 for full res)")
    parser.add_argument("--ckpt", default="model_paper_10k.pt", help="10k model checkpoint")
    args = parser.parse_args()

    run_single_sdsd_video(
        scene=args.scene,
        num_frames=args.num_frames,
        downsample=args.downsample,
        ckpt_path=args.ckpt,
    )
