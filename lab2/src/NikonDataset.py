from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, image_path: str = "./images", offset = 0, count = 0):
        self.images = sorted(
            [os.path.join(image_path, f) for f
             in os.listdir(image_path)
             if os.path.isfile(os.path.join(image_path, f))  and ".NEF" in f]
             )

        limit = offset+count
        if limit > len(self.images):
            limit = len(self.images)
        self.images = self.images[offset-1:self.images-1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean_img = self.slices[idx]
        nixon_img = self.nixon_Transform(image_img)
        transform = transforms.ToTensor()
        clean_transformed = transform(clean_img)
        nixon_transformed = transform(nixon_img)
        return clean_transformed, nixon_transformed

    def nixon_Transform(image):
        pass