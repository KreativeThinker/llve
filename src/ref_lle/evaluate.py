import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from .model import RefLLENet
from .infer import enhance_image
from .metrics import psnr, psnr_y, ssim, niqe, MetricTracker
from .fourier import amplitude_phase_split, zero_frequency_component


def evaluate_dataset(model, low_dir, gt_dir=None, zfc_target=2.5e5, max_iterations=10, device='cpu', output_dir=None):
    low_dir = Path(low_dir)
    gt_dir = Path(gt_dir) if gt_dir else None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    low_files = sorted(list(low_dir.glob("*.png")) + list(low_dir.glob("*.jpg")))
    tracker = MetricTracker()

    print(f"Evaluating {len(low_files)} images from {low_dir}...")
    for low_file in tqdm(low_files):
        low_img = Image.open(low_file).convert("RGB")
        low_arr = np.array(low_img, dtype=np.float32) / 255.0
        low_tensor = torch.from_numpy(np.transpose(low_arr, (2, 0, 1)))

        target_zfc = zfc_target
        gt_tensor = None
        if gt_dir and gt_dir.exists():
            gt_file = gt_dir / low_file.name
            if gt_file.exists():
                gt_img = Image.open(gt_file).convert("RGB")
                gt_arr = np.array(gt_img, dtype=np.float32) / 255.0
                gt_tensor = torch.from_numpy(np.transpose(gt_arr, (2, 0, 1)))
                # If evaluating reference-guided mode, target can match GT ZFC
                # ref_amp, _ = amplitude_phase_split(gt_tensor.unsqueeze(0))
                # target_zfc = zero_frequency_component(ref_amp).mean().item()

        enhanced = enhance_image(
            model=model,
            input_tensor=low_tensor,
            zfc_target=target_zfc,
            max_iterations=max_iterations,
            device=device,
            deterministic=True
        )

        if output_dir:
            Image.fromarray(enhanced).save(output_dir / low_file.name)

        enh_tensor = torch.from_numpy(np.transpose(enhanced.astype(np.float32) / 255.0, (2, 0, 1)))

        if gt_tensor is not None:
            tracker.add("PSNR", psnr(gt_tensor, enh_tensor))
            tracker.add("PSNR_y", psnr_y(gt_tensor, enh_tensor))
            tracker.add("SSIM", ssim(gt_tensor, enh_tensor))

        # No-reference quality
        tracker.add("NIQE", niqe(enh_tensor))

    summary = tracker.summary()
    print("\n--- Evaluation Results ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReF-LLE benchmark metrics (PSNR, PSNR_y, SSIM, NIQE)")
    parser.add_argument("--low-path", default="data/lolv2-real/Test/Input", type=str)
    parser.add_argument("--gt-path", default="data/lolv2-real/Test/GT", type=str)
    parser.add_argument("--model", default="model.pt", type=str)
    parser.add_argument("--zfc-ref", default=2.5e5, type=float)
    parser.add_argument("--max-iterations", default=10, type=int)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    hidden_dim = 32
    ckpt = None
    if Path(args.model).exists():
        ckpt = torch.load(args.model, map_location=args.device)
        if "encoder.conv1.weight" in ckpt:
            hidden_dim = ckpt["encoder.conv1.weight"].shape[0]

    model = RefLLENet(input_channels=3, hidden_dim=hidden_dim, num_actions=31)
    if ckpt is not None:
        model.load_state_dict(ckpt)
        print(f"Loaded weights from {args.model} (hidden_dim={hidden_dim})")
    else:
        print(f"Warning: Model checkpoint '{args.model}' not found, evaluating with initialized weights.")

    evaluate_dataset(
        model=model,
        low_dir=args.low_path,
        gt_dir=args.gt_path,
        zfc_target=args.zfc_ref,
        max_iterations=args.max_iterations,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
