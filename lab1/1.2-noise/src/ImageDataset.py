from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, image_path: str , noise_stddev):
        self.images = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
        self.images = self.images[0:200]
        self.noise_stddev = noise_stddev

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean_image = Image.open(self.images[idx]).convert("RGB").crop((0,0,256,256))
        transform = transforms.ToTensor()
        image_normal = transform(clean_image)
        noisy_img = self.ApplyNoise(image_normal)
        return image_normal, noisy_img

    def ApplyNoise(self, image):
        noise = torch.randn_like(image) * self.noise_stddev
        image_noise = (image + noise).clamp(0.0, 1.0)
        return image_noise