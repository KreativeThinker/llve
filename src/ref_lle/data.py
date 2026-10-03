import numpy as np
import torch
from PIL import Image
from pathlib import Path


def generate_synthetic_image(height=128, width=128):
    image = np.random.rand(height, width).astype(np.float32)
    return image


def darken_image(image, factor=0.3):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if image.ndim == 3:
        image = np.mean(image, axis=0)

    darkened = image * factor
    return np.clip(darkened, 0, 1).astype(np.float32)


class LOLDataset:
    def __init__(self, low_light_path=None, normal_light_path=None):
        self.low_light_path = Path(low_light_path) if low_light_path else None
        self.normal_light_path = Path(normal_light_path) if normal_light_path else None
        self.pairs = []

        if self.low_light_path and self.low_light_path.exists():
            self._load_lol_pairs()
        else:
            self._generate_synthetic_pairs()

    def _generate_synthetic_pairs(self):
        for i in range(10):
            normal = generate_synthetic_image(128, 128)
            low = darken_image(normal, factor=0.3)
            self.pairs.append((low, normal))

    def _load_lol_pairs(self):
        low_files = sorted(self.low_light_path.glob("*.png"))
        for low_file in low_files[:100]:
            normal_file = self.normal_light_path / low_file.name
            if normal_file.exists():
                low = np.array(Image.open(low_file)).astype(np.float32) / 255.0
                normal = np.array(Image.open(normal_file)).astype(np.float32) / 255.0

                if low.ndim == 3:
                    low = np.mean(low, axis=2)
                if normal.ndim == 3:
                    normal = np.mean(normal, axis=2)

                self.pairs.append((low, normal))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low, normal = self.pairs[idx]
        return torch.from_numpy(low), torch.from_numpy(normal)
