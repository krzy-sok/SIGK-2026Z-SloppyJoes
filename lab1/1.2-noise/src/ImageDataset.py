from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, image_path: str , noise_stddev):
        self.images = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
        self.images = self.images
        self.noise_stddev = noise_stddev
        self.slices= []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean_image: Image = Image.open(self.images[idx]).convert("RGB")
        # .crop((0,0,256,256))
        if len(self.slices) ==0:
            for y in range(0, clean_image.size[1] - 256, 256):
                for x in range(0, clean_image.size[0] - 256, 256):
                    self.slices.append(clean_image.crop((x,y,x+256,y+256)))
        # x = np.max((image.shape[1]/2)-256, 0)
        # y = np.max((image.shape[0]/2)-256, 0)
        clean_slice = self.slices.pop(0)
        transform = transforms.ToTensor()
        image_normal = transform(clean_slice)
        noisy_img = self.ApplyNoise(image_normal)
        return image_normal, noisy_img

    def ApplyNoise(self, image):
        noise = torch.randn_like(image) * self.noise_stddev
        image_noise = (image + noise).clamp(0.0, 1.0)
        return image_noise