from src.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os

IMG_PATH = "./lab1/1.2-noise/tests/test_images"


def test_split_test_images():
    dataset = ImageDataset(IMG_PATH, 0.03)
    assert(len(dataset) > 10)
    expected = 0
    images = sorted([os.path.join(IMG_PATH, f) for f in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, f))])
    for image in images:
        image = Image.open(image).convert("RGB")
        y = int(image.size[0] / 256)
        x = int(image.size[1] / 256)
        expected += x*y
    assert(len(dataset) == expected)

def test_noisy_not_original():
    dataset = ImageDataset(IMG_PATH, 0.03)
    for i in range(len(dataset)):
        # breakpoint()
        clean, noisy = dataset[i]
        assert(clean.size() == noisy.size())
        assert(not torch.equal(clean, noisy))