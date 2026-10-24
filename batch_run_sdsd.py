"""
Batch evaluate Motion-Consistent Temporal ReF-LLE across SDSD Indoor Video Scenes.
"""

import argparse
import time
import numpy as np
import torch
from ref_lle.infer_video import enhance_sdsd_video


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate ReF-LLE on SDSD video scenes.")
    parser.add_argument("--data-root", default=None, type=str, help="SDSD data root")
    parser.add_argument("--scenes", nargs="+", default=["pair2", "pair11", "pair12", "pair13", "pair14"], help="Scenes to evaluate")
    parser.add_argument("--num-frames", default=12, type=int)
    parser.add_argument("--downsample", default=2, type=int)
    parser.add_argument("--model-path", default="model_temporal_sdsd.pt", type=str)
    parser.add_argument("--output-dir", default="results/batch_sdsd", type=str)
    parser.add_argument("--steps-per-frame", default=10, type=int)
    parser.add_argument("--warmup-steps", default=14, type=int)
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    results = []
    print(f"Batch evaluating {len(args.scenes)} SDSD video scenes...")

    for idx, sc in enumerate(args.scenes, 1):
        t0 = time.time()
        print(f"\n[{idx}/{len(args.scenes)}] Evaluating scene: {sc}...")
        res = enhance_sdsd_video(
            data_root=args.data_root,
            scene=sc,
            num_frames=args.num_frames,
            downsample=args.downsample,
            model_path=args.model_path,
            output_dir=args.output_dir,
            steps_per_frame=args.steps_per_frame,
            warmup_steps=args.warmup_steps,
            fps=args.fps,
            device=args.device,
        )
        dt = time.time() - t0
        res["elapsed_s"] = dt
        results.append(res)

    print("\n" + "=" * 90)
    print(f"{'Scene':<10} | {'Brightness (In -> Out)':<24} | {'PSNR (In -> Out)':<24} | {'SSIM (In -> Out)':<24}")
    print("=" * 90)
    for r in results:
        b_str = f"{r['raw_in_mean']:.3f} -> {r['enhanced_mean']:.3f}"
        p_str = f"{r.get('psnr_in', 0.0):.2f} -> {r.get('psnr_enh', 0.0):.2f} dB"
        s_str = f"{r.get('ssim_in', 0.0):.4f} -> {r.get('ssim_enh', 0.0):.4f}"
        print(f"{r['scene']:<10} | {b_str:<24} | {p_str:<24} | {s_str:<24}")
    print("=" * 90)

    if any("psnr_enh" in r for r in results):
        avg_psnr_gain = np.mean([r["psnr_enh"] - r["psnr_in"] for r in results if "psnr_enh" in r])
        avg_ssim_gain = np.mean([r["ssim_enh"] - r["ssim_in"] for r in results if "ssim_enh" in r])
        print(f"\nOverall Mean PSNR Gain: +{avg_psnr_gain:.2f} dB")
        print(f"Overall Mean SSIM Gain: +{avg_ssim_gain:.4f}")


if __name__ == "__main__":
    main()
