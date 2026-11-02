from src.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from src.Unet import Unet

IMG_PATH = "./lab1/1.2-noise/tests/test_images"


def test_model_output_size():
    dataset = ImageDataset(IMG_PATH, 0.03)
    batch_size = 1
    loader = DataLoader(dataset, batch_size=batch_size)
    model = Unet()
    noisy_img = next(iter(loader))[1]
    out_img = model(noisy_img.to("cpu"))

    assert(torch.squeeze(out_img).size() == dataset[0][1].size())
    assert(out_img.size() == torch.Size([batch_size,3,256,256]))
