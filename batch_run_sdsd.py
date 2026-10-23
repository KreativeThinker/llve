"""
Batch evaluate Causal Temporal Difference ReF-LLE on 5 SDSD Video Scenes.
Scenes: pair2, pair11, pair12, pair13, pair14
"""

import time
from pathlib import Path
import numpy as np
import torch
import imageio
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim

from ref_lle.model import TemporalDiffRefLLENet
from ref_lle.fourier import amplitude_phase_split, recombine, zero_frequency_component
from ref_lle.video_data import load_frame_npy_or_img
from ref_lle.infer_video import compute_flicker_index, compute_temporal_consistency_error


def evaluate_scene(
    scene_name,
    model,
    num_frames=12,
    downsample=2,
    steps_per_frame=4,
    device="cpu",
    output_dir="data/sdsd",
):
    output_dir = Path(output_dir)
    in_dir = Path(f"data/sdsd/indoor_np/indoor_np/input/{scene_name}")
    gt_dir = Path(f"data/sdsd/indoor_np/indoor_np/GT/{scene_name}")

    in_files = sorted(in_dir.glob("*.npy"))[:num_frames]
    gt_files = sorted(gt_dir.glob("*.npy"))[:num_frames]

    raw_in = [load_frame_npy_or_img(f, downsample=downsample) for f in in_files]
    raw_in = [np.transpose(f, (1, 2, 0)) for f in raw_in]  # (H, W, 3)

    raw_gt = [load_frame_npy_or_img(f, downsample=downsample) for f in gt_files]
    raw_gt = [np.transpose(f, (1, 2, 0)) for f in raw_gt]

    T = len(raw_in)
    H, W, C = raw_in[0].shape

    # Target ZFC from Ground Truth
    ref_t = torch.from_numpy(np.transpose(raw_gt[0], (2, 0, 1))).unsqueeze(0).float().to(device)
    ref_amp, _ = amplitude_phase_split(ref_t)
    zfc_target = zero_frequency_component(ref_amp).mean().item()

    enhanced_frames = []
    prev_enhanced_img = None
    prev_enhanced_amp = None
    prev_gain = None

    alpha_min, alpha_max, alpha_step = -0.1, 0.35, 0.015

    for t in range(T):
        cur_img = raw_in[t]
        t_cur = torch.from_numpy(np.transpose(cur_img, (2, 0, 1))).unsqueeze(0).float().to(device)
        amp_0, pha_0 = amplitude_phase_split(t_cur)
        amp_cur = amp_0.clone()

        if prev_enhanced_img is None:
            prev_enhanced_img = t_cur.clone()
            prev_enhanced_amp = amp_0.clone()
            delta_prev_updated = torch.zeros_like(t_cur)
        else:
            delta_prev_updated = t_cur - prev_enhanced_img

        cum_gain = torch.ones_like(amp_cur)
        steps = steps_per_frame if t == 0 else max(2, steps_per_frame - 1)

        for step in range(steps):
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
            step_gain = torch.exp(alphas).unsqueeze(1)

            amp_cur = amp_cur * step_gain
            cum_gain = cum_gain * step_gain

            img_step = recombine(amp_cur, pha_0)
            img_step = torch.clamp(img_step, 0.0, 1.0)

            zfc_cur = zero_frequency_component(amp_cur).mean().item()
            if abs(zfc_cur - zfc_target) / zfc_target < 0.08:
                break

        final_frame = np.transpose(img_step.squeeze(0).cpu().numpy(), (1, 2, 0))
        enhanced_frames.append(final_frame)

        prev_enhanced_img = img_step.detach().clone()
        prev_enhanced_amp = amp_cur.detach().clone()
        prev_gain = cum_gain.detach().clone()

    # Save output animations
    out_gif = output_dir / f"{scene_name}_enhanced.gif"
    enh_uint8 = [np.clip(f * 255.0, 0, 255).astype(np.uint8) for f in enhanced_frames]
    imageio.mimsave(str(out_gif), enh_uint8, fps=6)

    cmp_gif = output_dir / f"{scene_name}_comparison.gif"
    cmp_frames = [
        np.hstack([
            np.clip(i * 255.0, 0, 255).astype(np.uint8),
            np.clip(e * 255.0, 0, 255).astype(np.uint8),
            np.clip(g * 255.0, 0, 255).astype(np.uint8),
        ])
        for i, e, g in zip(raw_in, enhanced_frames, raw_gt)
    ]
    imageio.mimsave(str(cmp_gif), cmp_frames, fps=6)

    # Compute metrics
    in_bright = float(np.mean(raw_in))
    enh_bright = float(np.mean(enhanced_frames))
    gt_bright = float(np.mean(raw_gt))

    in_contrast = float(np.std(raw_in))
    enh_contrast = float(np.std(enhanced_frames))

    psnr_in = float(np.mean([compute_psnr(g, i, data_range=1.0) for g, i in zip(raw_gt, raw_in)]))
    psnr_enh = float(np.mean([compute_psnr(g, e, data_range=1.0) for g, e in zip(raw_gt, enhanced_frames)]))

    ssim_in = float(np.mean([compute_ssim(g, i, channel_axis=2, data_range=1.0) for g, i in zip(raw_gt, raw_in)]))
    ssim_enh = float(np.mean([compute_ssim(g, e, channel_axis=2, data_range=1.0) for g, e in zip(raw_gt, enhanced_frames)]))

    flicker_in = compute_flicker_index(raw_in)
    flicker_enh = compute_flicker_index(enhanced_frames)

    return {
        "scene": scene_name,
        "in_bright": in_bright,
        "enh_bright": enh_bright,
        "gt_bright": gt_bright,
        "in_contrast": in_contrast,
        "enh_contrast": enh_contrast,
        "psnr_in": psnr_in,
        "psnr_enh": psnr_enh,
        "psnr_diff": psnr_enh - psnr_in,
        "ssim_in": ssim_in,
        "ssim_enh": ssim_enh,
        "ssim_diff": ssim_enh - ssim_in,
        "flicker_in": flicker_in,
        "flicker_enh": flicker_enh,
    }


def main():
    device = torch.device("cpu")
    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31).to(device)
    ckpt = torch.load("model_temporal_sdsd.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("Loaded model_temporal_sdsd.pt successfully.")

    scenes = ["pair2", "pair11", "pair12", "pair13", "pair14"]
    results = []

    print(f"\nProcessing {len(scenes)} SDSD video scenes...")
    for idx, sc in enumerate(scenes, 1):
        t0 = time.time()
        print(f"\n[{idx}/{len(scenes)}] Evaluating {sc}...")
        res = evaluate_scene(sc, model, num_frames=12, downsample=2, device=device)
        dt = time.time() - t0
        print(f"  {sc} done in {dt:.1f}s | PSNR: {res['psnr_in']:.2f} -> {res['psnr_enh']:.2f} dB (+{res['psnr_diff']:.2f} dB) | SSIM: {res['ssim_in']:.4f} -> {res['ssim_enh']:.4f}")
        results.append(res)

    print("\n" + "=" * 95)
    print(f"{'Scene':<10} | {'Brightness (In -> Enh)':<24} | {'PSNR Before -> After':<24} | {'SSIM Before -> After':<24}")
    print("=" * 95)
    for r in results:
        b_str = f"{r['in_bright']:.3f} -> {r['enh_bright']:.3f} (GT: {r['gt_bright']:.2f})"
        p_str = f"{r['psnr_in']:.2f} -> {r['psnr_enh']:.2f} dB (+{r['psnr_diff']:.2f})"
        s_str = f"{r['ssim_in']:.4f} -> {r['ssim_enh']:.4f} (+{r['ssim_diff']:.4f})"
        print(f"{r['scene']:<10} | {b_str:<24} | {p_str:<24} | {s_str:<24}")
    print("=" * 95)

    avg_psnr_gain = np.mean([r['psnr_diff'] for r in results])
    avg_ssim_gain = np.mean([r['ssim_diff'] for r in results])
    print(f"\nOverall Mean PSNR Gain: +{avg_psnr_gain:.2f} dB")
    print(f"Overall Mean SSIM Gain: +{avg_ssim_gain:.4f}")


if __name__ == "__main__":
    main()
