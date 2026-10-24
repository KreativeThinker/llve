"""
Run Temporal Difference ReF-LLE on SDSD Video Dataset.
Thin wrapper delegating to ref_lle.infer_video.enhance_sdsd_video.
"""

import argparse
import torch
from ref_lle.infer_video import enhance_sdsd_video


def main():
    parser = argparse.ArgumentParser(description="Enhance low-light video with motion-consistent temporal consistency.")
    parser.add_argument("--data-root", default=None, type=str, help="Path to SDSD dataset root directory")
    parser.add_argument("--scene", default="pair1", type=str)
    parser.add_argument("--num-frames", default=12, type=int)
    parser.add_argument("--downsample", default=2, type=int)
    parser.add_argument("--model-path", default="model_temporal_sdsd.pt", type=str)
    parser.add_argument("--output-dir", default="data/sdsd", type=str)
    parser.add_argument("--steps-per-frame", default=10, type=int)
    parser.add_argument("--warmup-steps", default=14, type=int)
    parser.add_argument("--fps", default=10, type=int, help="Frames per second for output MP4 videos")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
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
