import numpy as np
import torch
from PIL import Image
from pathlib import Path
import random


def generate_synthetic_image(height=128, width=128, channels=3):
    return np.random.rand(channels, height, width).astype(np.float32)


def darken_image(image, factor=0.3):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    darkened = image * factor
    return np.clip(darkened, 0, 1).astype(np.float32)


class LOLDataset:
    """Dataset loader for LOL / LOL-v2 / custom pairs with on-the-fly loading."""
    def __init__(self, low_light_path=None, normal_light_path=None, patch_size=None, is_train=True, max_samples=None):
        self.low_light_path = Path(low_light_path) if low_light_path else None
        self.normal_light_path = Path(normal_light_path) if normal_light_path else None
        self.patch_size = patch_size
        self.is_train = is_train
        self.pairs = []

        if self.low_light_path and self.low_light_path.exists():
            self._load_file_pairs(max_samples)
        else:
            self._generate_synthetic_pairs(num_pairs=max_samples or 20)

    def _generate_synthetic_pairs(self, num_pairs=20):
        self.is_synthetic = True
        self.synthetic_pairs = []
        for _ in range(num_pairs):
            normal = generate_synthetic_image(128, 128, 3)
            low = darken_image(normal, factor=0.3)
            self.synthetic_pairs.append((low, normal))

    def _load_file_pairs(self, max_samples=None):
        self.is_synthetic = False
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        low_files = []
        for ext in extensions:
            low_files.extend(list(self.low_light_path.glob(ext)))
        low_files = sorted(low_files)

        for low_file in low_files:
            if self.normal_light_path and self.normal_light_path.exists():
                normal_file = self.normal_light_path / low_file.name
                if not normal_file.exists():
                    # Check with same stem in normal directory
                    candidates = list(self.normal_light_path.glob(f"{low_file.stem}.*"))
                    normal_file = candidates[0] if candidates else None
            else:
                normal_file = None

            if normal_file and normal_file.exists():
                self.pairs.append((low_file, normal_file))
            elif not self.normal_light_path:
                # Unpaired mode (e.g. testing on DICM/LIME/MEF)
                self.pairs.append((low_file, low_file))

            if max_samples and len(self.pairs) >= max_samples:
                break

    def __len__(self):
        if getattr(self, 'is_synthetic', False):
            return len(self.synthetic_pairs)
        return len(self.pairs)

    def __getitem__(self, idx):
        if getattr(self, 'is_synthetic', False):
            low, normal = self.synthetic_pairs[idx]
            return torch.from_numpy(low), torch.from_numpy(normal)

        low_path, norm_path = self.pairs[idx]
        low_img = Image.open(low_path).convert("RGB")
        norm_img = Image.open(norm_path).convert("RGB") if norm_path else low_img

        low_arr = np.array(low_img, dtype=np.float32) / 255.0
        norm_arr = np.array(norm_img, dtype=np.float32) / 255.0

        # Optional random crop for training stability
        if self.patch_size is not None:
            h, w, _ = low_arr.shape
            ps = self.patch_size
            if h >= ps and w >= ps:
                if self.is_train:
                    top = random.randint(0, h - ps)
                    left = random.randint(0, w - ps)
                else:
                    top = (h - ps) // 2
                    left = (w - ps) // 2
                low_arr = low_arr[top:top + ps, left:left + ps]
                norm_arr = norm_arr[top:top + ps, left:left + ps]

        # Convert to CHW tensor
        low_tensor = torch.from_numpy(np.transpose(low_arr, (2, 0, 1)))
        norm_tensor = torch.from_numpy(np.transpose(norm_arr, (2, 0, 1)))
        return low_tensor, norm_tensor
