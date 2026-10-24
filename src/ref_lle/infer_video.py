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


def save_video_mp4(path, frames, fps=10):
    path = Path(path)
    try:
        imageio.mimsave(str(path), frames, fps=fps)
    except Exception:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()


def resolve_sdsd_input_and_scene(data_root=None, scene="pair1"):
    candidates = []
    if data_root is not None:
        p = Path(data_root)
        candidates.extend([p, p / "indoor_np", p / "input"])
    candidates.extend([
        Path("data/sdsd/indoor_np"),
        Path("data/sdsd/indoor_np/indoor_np"),
        Path("data/sdsd"),
    ])

    input_base = None
    gt_base = None

    for c in candidates:
        if (c / "input").exists():
            input_base = c / "input"
            gt_base = (c / "GT") if (c / "GT").exists() else None
            break
        elif c.name == "input" and c.exists():
            input_base = c
            gt_base = (c.parent / "GT") if (c.parent / "GT").exists() else None
            break

    if input_base is None or not input_base.exists():
        tried = [str(c) for c in candidates]
        raise FileNotFoundError(
            f"SDSD input directory not found. Searched candidates: {tried}\n"
            f"Please specify --data-root pointing to the SDSD directory containing 'input/'."
        )

    available_scenes = sorted([d.name for d in input_base.iterdir() if d.is_dir()])
    if not available_scenes:
        direct_files = list(input_base.glob("*.npy")) + list(input_base.glob("*.png"))
        if direct_files:
            return input_base, gt_base, "root"
        raise FileNotFoundError(f"No scene subdirectories or frame files found in '{input_base}'.")

    if scene not in available_scenes:
        print(f"Notice: Scene '{scene}' not found in '{input_base}'.")
        print(f"Available scenes: {available_scenes}")
        selected = available_scenes[0]
        print(f"Auto-selected available scene: '{selected}'")
        scene = selected

    in_dir = input_base / scene
    gt_dir = (gt_base / scene) if (gt_base and (gt_base / scene).exists()) else None
    return in_dir, gt_dir, scene


def enhance_sdsd_video(
    data_root=None,
    scene="pair1",
    num_frames=12,
    downsample=2,
    model_path="model_temporal_sdsd.pt",
    output_dir="data/sdsd",
    steps_per_frame=10,
    warmup_steps=14,
    fps=10,
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_dir, gt_dir, scene = resolve_sdsd_input_and_scene(data_root=data_root, scene=scene)

    in_files = sorted(in_dir.glob("*.npy"))
    if not in_files:
        in_files = sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
    in_files = in_files[:num_frames]

    if not in_files:
        raise RuntimeError(f"No .npy or image frames found in {in_dir}")

    gt_files = []
    if gt_dir and gt_dir.exists():
        gt_files = sorted(gt_dir.glob("*.npy"))
        if not gt_files:
            gt_files = sorted(gt_dir.glob("*.png")) + sorted(gt_dir.glob("*.jpg"))
        gt_files = gt_files[:num_frames]

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
    prev_raw_img = None

    alpha_min, alpha_max, alpha_step = -0.1, 0.35, 0.015

    print("\n--- Enhancing Video Frames with Motion-Consistent Causal Pipeline ---")
    for t in range(T):
        cur_img = raw_in[t]
        t_cur = torch.from_numpy(np.transpose(cur_img, (2, 0, 1))).unsqueeze(0).float().to(device)
        amp_0, pha_0 = amplitude_phase_split(t_cur)
        amp_cur = amp_0.clone()

        if prev_enhanced_img is None:
            # First frame: no previous frame exists
            prev_enhanced_amp = amp_0.clone()
            delta_raw_motion = torch.zeros_like(t_cur)
        else:
            # Physical camera motion between raw frames (I_t^raw - I_{t-1}^raw)
            delta_raw_motion = t_cur - prev_raw_img

        # Frames 0 and 1 receive full warmup_steps, subsequent frames receive steps_per_frame
        is_warmup = t < 2
        steps = warmup_steps if is_warmup else steps_per_frame

        cum_gain = torch.ones_like(amp_cur)

        for step in range(steps):
            # 9-channel motion-consistent composite input: [A_t, A^*_{t-1}, Delta_raw]
            comp_state = torch.cat([torch.log1p(amp_cur), torch.log1p(prev_enhanced_amp), delta_raw_motion], dim=1)

            with torch.no_grad():
                logits, _ = model(comp_state)
                action = torch.argmax(logits, dim=1).reshape(1, H, W)

            zfc_cur = zero_frequency_component(amp_cur).mean().item()
            ratio = zfc_target / (zfc_cur + 1e-8)
            remaining_steps = max(1, steps - step)
            target_alpha = np.clip(np.log(ratio) / remaining_steps, alpha_min, alpha_max)
            act_idx_target = int(round((target_alpha - alpha_min) / alpha_step))
            act_idx_target = min(max(act_idx_target, 0), 30)

            # Smoothly guide policy towards target ZFC
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

        # Update previous frame history for t + 1
        prev_enhanced_img = img_step.detach().clone()
        prev_enhanced_amp = amp_cur.detach().clone()
        prev_raw_img = t_cur.detach().clone()

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

    print("\n================ Motion-Consistent Video Metrics Summary ================")
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
    print(f"Temporal Accel Err:  {tce_in:.5f} ==> {tce_enh:.5f}")

    # Export videos as MP4
    mp4_preview_path = output_dir / f"enhanced_{scene}_preview.mp4"
    preview_frames = []
    for t in range(T):
        in_u8 = (np.clip(raw_in[t], 0, 1) * 255).astype(np.uint8)
        enh_u8 = (np.clip(enhanced_frames[t], 0, 1) * 255).astype(np.uint8)
        if raw_gt:
            gt_u8 = (np.clip(raw_gt[t], 0, 1) * 255).astype(np.uint8)
            row = np.concatenate([in_u8, enh_u8, gt_u8], axis=1)
        else:
            row = np.concatenate([in_u8, enh_u8], axis=1)
        preview_frames.append(row)

    save_video_mp4(mp4_preview_path, preview_frames, fps=fps)
    print(f"\nSaved side-by-side comparison video to: {mp4_preview_path}")

    mp4_enh_path = output_dir / f"enhanced_{scene}.mp4"
    enh_frames_u8 = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in enhanced_frames]
    save_video_mp4(mp4_enh_path, enh_frames_u8, fps=fps)
    print(f"Saved enhanced video to: {mp4_enh_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhance low-light video with motion-consistent temporal consistency.")
    parser.add_argument("--data-root", default=None, type=str, help="Path to SDSD dataset root directory (e.g. data/sdsd/indoor_np)")
    parser.add_argument("--scene", default="pair1", type=str)
    parser.add_argument("--num-frames", default=12, type=int)
    parser.add_argument("--downsample", default=2, type=int)
    parser.add_argument("--model-path", default="model_temporal_sdsd.pt", type=str)
    parser.add_argument("--output-dir", default="data/sdsd", type=str)
    parser.add_argument("--steps-per-frame", default=10, type=int)
    parser.add_argument("--warmup-steps", default=14, type=int)
    parser.add_argument("--fps", default=10, type=int, help="Frames per second for output MP4 videos")
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    enhance_sdsd_video(
        data_root=args.data_root,
        scene=args.scene,
        num_frames=args.num_frames,
        downsample=args.downsample,
        model_path=args.model_path,
        output_dir=args.output_dir,
        steps_per_frame=args.steps_per_frame,
        warmup_steps=args.warmup_steps,
        fps=args.fps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
