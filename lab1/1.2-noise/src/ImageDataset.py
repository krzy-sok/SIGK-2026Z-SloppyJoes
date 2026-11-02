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
        self.prepare_images()

    def __len__(self):
          return len(self.slices)
        # return len(self.images)

    def __getitem__(self, idx):
        # clean_image: Image = Image.open(self.images[idx]).convert("RGB")
        # .crop((0,0,256,256))
        clean_slice = self.slices[idx]
        transform = transforms.ToTensor()
        image_normal = transform(clean_slice)
        noisy_img = self.ApplyNoise(image_normal)
        return image_normal, noisy_img

    def ApplyNoise(self, image):
        noise = torch.randn_like(image) * self.noise_stddev
        image_noise = (image + noise).clamp(0.0, 1.0)
        return image_noise

    def prepare_images(self):
        for i, image in enumerate(self.images):
            len_prev = len(self.slices)
            image = Image.open(image).convert("RGB")
            for y in range(0, image.size[1] - 256, 256):
                for x in range(0, image.size[0] - 256, 256):
                    self.slices.append(image.crop((x,y,x+256,y+256)))
            len_new = len(self.slices)
            # print(f"sliced image {i} into {len_new-len_prev}")
