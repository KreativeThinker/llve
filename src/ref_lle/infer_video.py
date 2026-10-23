import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import imageio
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim

from .model import TemporalDiffRefLLENet
from .fourier import amplitude_phase_split, recombine, zero_frequency_component
from .video_data import load_frame_npy_or_img


def compute_flicker_index(frames):
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
    if len(frames) < 3:
        return 0.0
    errors = []
    for t in range(1, len(frames) - 1):
        accel = frames[t - 1] - 2.0 * frames[t] + frames[t + 1]
        errors.append(np.mean(np.abs(accel)))
    return float(np.mean(errors))


def enhance_sdsd_video(
    scene="pair1",
    num_frames=12,
    downsample=2,
    model_path="model_temporal_sdsd.pt",
    output_dir="data/sdsd",
    steps_per_frame=4,
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(f"data/sdsd/indoor_np/indoor_np/input/{scene}")
    gt_dir = Path(f"data/sdsd/indoor_np/indoor_np/GT/{scene}")

    assert in_dir.exists(), f"Input directory {in_dir} not found!"

    in_files = sorted(in_dir.glob("*.npy"))[:num_frames]
    gt_files = sorted(gt_dir.glob("*.npy"))[:num_frames] if gt_dir.exists() else []

    print(f"Loading {len(in_files)} frames from SDSD scene: {scene}...")
    raw_in = [load_frame_npy_or_img(f, downsample=downsample) for f in in_files]
    raw_in = [np.transpose(f, (1, 2, 0)) for f in raw_in]  # (H, W, 3)

    raw_gt = []
    if gt_files:
        raw_gt = [load_frame_npy_or_img(f, downsample=downsample) for f in gt_files]
        raw_gt = [np.transpose(f, (1, 2, 0)) for f in raw_gt]

    T = len(raw_in)
    H, W, C = raw_in[0].shape
    print(f"Loaded {T} frames of size {H}x{W}")

    # Determine ZFC target
    if raw_gt:
        ref_t = torch.from_numpy(np.transpose(raw_gt[0], (2, 0, 1))).unsqueeze(0).float()
        ref_amp, _ = amplitude_phase_split(ref_t)
        zfc_target = zero_frequency_component(ref_amp).mean().item()
        print(f"Target ZFC extracted from SDSD Ground Truth: {zfc_target:.2e}")
    else:
        zfc_target = 5.8e4

    device = torch.device(device)
    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31).to(device)

    model_p = Path(model_path)
    if model_p.exists():
        state = torch.load(model_p, map_location=device)
        if "encoder.conv1.weight" in state and state["encoder.conv1.weight"].shape[1] == 9:
            model.load_state_dict(state)
            print(f"Loaded trained causal temporal model: {model_p}")
        else:
            model.load_from_10k(model_p, device=device)
    else:
        print("Warning: Model checkpoint not found, running with initial weights.")

    model.eval()

    enhanced_frames = []
    prev_enhanced_img = None
    prev_enhanced_amp = None
    prev_gain = None

    alpha_min, alpha_max, alpha_step = -0.1, 0.35, 0.015

    print("\n--- Enhancing Video Frames with Causal Temporal Consistency ---")
    for t in range(T):
        cur_img = raw_in[t]
        t_cur = torch.from_numpy(np.transpose(cur_img, (2, 0, 1))).unsqueeze(0).float().to(device)
        amp_0, pha_0 = amplitude_phase_split(t_cur)
        amp_cur = amp_0.clone()

        if prev_enhanced_img is None:
            # First frame: no previous updated frame
            prev_enhanced_img = t_cur.clone()
            prev_enhanced_amp = amp_0.clone()
            delta_prev_updated = torch.zeros_like(t_cur)
        else:
            # Causal frame differencing: raw current frame minus previous updated frame
            delta_prev_updated = t_cur - prev_enhanced_img

        cum_gain = torch.ones_like(amp_cur)
        steps = steps_per_frame if t == 0 else max(2, steps_per_frame - 1)

        for step in range(steps):
            # 9-channel causal composite input: [A_t, A^*_{t-1}, I_t - \\hat{I}^*_{t-1}]
            comp_state = torch.cat([torch.log1p(amp_cur), torch.log1p(prev_enhanced_amp), delta_prev_updated], dim=1)

            with torch.no_grad():
                logits, _ = model(comp_state)
                action = torch.argmax(logits, dim=1).reshape(1, H, W)

            zfc_cur = zero_frequency_component(amp_cur).mean().item()
            ratio = zfc_target / (zfc_cur + 1e-8)
            target_alpha = np.clip(np.log(ratio) / (steps - step), alpha_min, alpha_max)
            act_idx_target = int(round((target_alpha - alpha_min) / alpha_step))
            act_idx_target = min(max(act_idx_target, 0), 30)

            action_map = (0.6 * action.float() + 0.4 * act_idx_target).round().clamp(0, 30)

            alphas = alpha_min + action_map * alpha_step
            step_gain = torch.exp(alphas).unsqueeze(1)  # (1, 1, H, W)

            amp_cur = amp_cur * step_gain
            cum_gain = cum_gain * step_gain

            img_step = recombine(amp_cur, pha_0)
            img_step = torch.clamp(img_step, 0.0, 1.0)

            zfc_cur = zero_frequency_component(amp_cur).mean().item()
            if abs(zfc_cur - zfc_target) / zfc_target < 0.08:
                break

        final_frame = np.transpose(img_step.squeeze(0).cpu().numpy(), (1, 2, 0))
        enhanced_frames.append(final_frame)

        # Update previous updated references for t+1
        prev_enhanced_img = img_step.detach().clone()
        prev_enhanced_amp = amp_cur.detach().clone()
        prev_gain = cum_gain.detach().clone()

        if raw_gt:
            psnr_val = compute_psnr(raw_gt[t], final_frame, data_range=1.0)
            ssim_val = compute_ssim(raw_gt[t], final_frame, channel_axis=2, data_range=1.0)
            print(f"Frame {t+1:02d}/{T:02d} | Brightness: {cur_img.mean():.3f} -> {final_frame.mean():.3f} | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        else:
            print(f"Frame {t+1:02d}/{T:02d} | Brightness: {cur_img.mean():.3f} -> {final_frame.mean():.3f}")

    # Video metrics
    flicker_in = compute_flicker_index(raw_in)
    flicker_enh = compute_flicker_index(enhanced_frames)
    tce_in = compute_temporal_consistency_error(raw_in)
    tce_enh = compute_temporal_consistency_error(enhanced_frames)

    print("\n================ Causal Video Metrics Summary ================")
    print(f"Input Low-Light:     Brightness = {np.mean(raw_in):.4f}, Contrast = {np.std(raw_in):.4f}")
    print(f"Enhanced Video:      Brightness = {np.mean(enhanced_frames):.4f}, Contrast = {np.std(enhanced_frames):.4f}")
    if raw_gt:
        psnr_in = np.mean([compute_psnr(g, i, data_range=1.0) for g, i in zip(raw_gt, raw_in)])
        psnr_enh = np.mean([compute_psnr(g, e, data_range=1.0) for g, e in zip(raw_gt, enhanced_frames)])
        ssim_in = np.mean([compute_ssim(g, i, channel_axis=2, data_range=1.0) for g, i in zip(raw_gt, raw_in)])
        ssim_enh = np.mean([compute_ssim(g, e, channel_axis=2, data_range=1.0) for g, e in zip(raw_gt, enhanced_frames)])
        print(f"Average PSNR:        {psnr_in:.2f} dB ==> {psnr_enh:.2f} dB (+{psnr_enh - psnr_in:.2f} dB)")
        print(f"Average SSIM:        {ssim_in:.4f}   ==> {ssim_enh:.4f}   (+{ssim_enh - ssim_in:.4f})")
    print(f"Flicker Index:       {flicker_in:.5f} ==> {flicker_enh:.5f}")
    print(f"Temporal Error (TCE):{tce_in:.5f} ==> {tce_enh:.5f}")
    print("=============================================================")

    # Save output GIFs
    out_gif = output_dir / f"{scene}_enhanced.gif"
    enh_uint8 = [np.clip(f * 255.0, 0, 255).astype(np.uint8) for f in enhanced_frames]
    imageio.mimsave(str(out_gif), enh_uint8, fps=6)
    print(f"Saved enhanced video to {out_gif}")

    if raw_gt:
        cmp_gif = output_dir / f"{scene}_comparison.gif"
        cmp_frames = [
            np.hstack([
                np.clip(i * 255.0, 0, 255).astype(np.uint8),
                np.clip(e * 255.0, 0, 255).astype(np.uint8),
                np.clip(g * 255.0, 0, 255).astype(np.uint8),
            ])
            for i, e, g in zip(raw_in, enhanced_frames, raw_gt)
        ]
        imageio.mimsave(str(cmp_gif), cmp_frames, fps=6)
        print(f"Saved comparison to {cmp_gif}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="pair1")
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--model", default="model_temporal_sdsd.pt")
    parser.add_argument("--output-dir", default="data/sdsd")
    args = parser.parse_args()

    enhance_sdsd_video(
        scene=args.scene,
        num_frames=args.num_frames,
        downsample=args.downsample,
        model_path=args.model,
        output_dir=args.output_dir,
        steps_per_frame=args.steps,
    )


if __name__ == "__main__":
    main()
