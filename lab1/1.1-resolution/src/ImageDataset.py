import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.ResolutionScaler import ResolutionScaler

class ImageDataset(Dataset):
    def __init__(self, image_path: str, low_res_size: int = 32, max_images: int | None = 200):
        self.images = sorted(
            [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if os.path.isfile(os.path.join(image_path, f))
            ]
        )
        if max_images is not None:
            self.images = self.images[:max_images]
        self.low_res_size = low_res_size
        self.scaler = ResolutionScaler(target_size=low_res_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean_pil = Image.open(self.images[idx]).convert("RGB").crop((0, 0, 256, 256))
        clean_np = np.asarray(clean_pil, dtype=np.float32) / 255.0

        low_res_np = self.apply_downscale(clean_np)

        clean_tensor = torch.from_numpy(clean_np).permute(2, 0, 1).contiguous()
        low_res_tensor = torch.from_numpy(low_res_np).permute(2, 0, 1).contiguous()

        return low_res_tensor, clean_tensor

    def apply_downscale(self, image: np.ndarray) -> np.ndarray:
        downscaled = self.scaler.downscale_single(image)
        return np.clip(downscaled, 0.0, 1.0).astype(np.float32)
