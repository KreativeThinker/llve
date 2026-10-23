import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_frame_npy_or_img(path, downsample=1):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.dtype == np.uint8 or arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            if downsample > 1:
                arr = arr[::downsample, ::downsample]
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2:
            if downsample > 1:
                arr = arr[::downsample, ::downsample]
            arr = np.repeat(np.expand_dims(arr, 0), 3, axis=0)
        return arr
    else:
        img = Image.open(path).convert("RGB")
        if downsample > 1:
            w, h = img.size
            img = img.resize((w // downsample, h // downsample), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))


class SDSDVideoDataset:
    """
    In-memory accelerated dataset for SDSD indoor video clips.
    Loads and caches clips of length clip_len in RAM to allow fast multi-epoch training.
    """
    def __init__(
        self,
        input_dir="data/sdsd/indoor_np/indoor_np/input",
        gt_dir="data/sdsd/indoor_np/indoor_np/GT",
        clip_len=4,
        downsample=4,
        max_scenes=10,
        preload_to_ram=True,
    ):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.clip_len = clip_len
        self.downsample = downsample
        self.preload_to_ram = preload_to_ram

        self.cached_clips = []
        if self.input_dir.exists() and self.gt_dir.exists():
            scenes = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
            if max_scenes is not None:
                scenes = scenes[:max_scenes]

            for sdir in scenes:
                in_files = sorted(sdir.glob("*.npy"))
                gt_sdir = self.gt_dir / sdir.name
                if not gt_sdir.exists() or len(in_files) < self.clip_len:
                    continue

                for start_idx in range(0, min(len(in_files) - self.clip_len + 1, 16), self.clip_len):
                    in_clip_files = in_files[start_idx : start_idx + self.clip_len]
                    gt_clip_files = [gt_sdir / f.name for f in in_clip_files]
                    if all(g.exists() for g in gt_clip_files):
                        if self.preload_to_ram:
                            in_frames = [load_frame_npy_or_img(f, self.downsample) for f in in_clip_files]
                            gt_frames = [load_frame_npy_or_img(f, self.downsample) for f in gt_clip_files]
                            in_t = torch.from_numpy(np.stack(in_frames, axis=0))
                            gt_t = torch.from_numpy(np.stack(gt_frames, axis=0))
                            self.cached_clips.append((in_t, gt_t))
                        else:
                            self.cached_clips.append((in_clip_files, gt_clip_files))

        if not self.cached_clips:
            print("Warning: No SDSD clips found. Creating fallback clip.")
            in_t = torch.rand(self.clip_len, 3, 128, 128) * 0.2
            gt_t = torch.clamp(in_t * 3.0, 0, 1)
            self.cached_clips.append((in_t, gt_t))

    def __len__(self):
        return len(self.cached_clips)

    def __getitem__(self, idx):
        item = self.cached_clips[idx % len(self.cached_clips)]
        if self.preload_to_ram:
            return item
        else:
            in_files, gt_files = item
            in_frames = [load_frame_npy_or_img(f, self.downsample) for f in in_files]
            gt_frames = [load_frame_npy_or_img(f, self.downsample) for f in gt_files]
            return torch.from_numpy(np.stack(in_frames, axis=0)), torch.from_numpy(np.stack(gt_frames, axis=0))
