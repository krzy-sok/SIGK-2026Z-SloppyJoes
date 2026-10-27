from torch.utils.data import Dataset
from src.ImageLoader import ImageLoader
from src.NoiseApplier import NoiseApplier
import torch

class ImageDataset(Dataset):
    def __init__(self, image_path, noise_stddev):
        self.Loader = ImageLoader(0,0)
        self.Loader.load_from_folder(image_path)
        self.images_clean = self.Loader.images
        self.Noiser = NoiseApplier(noise_stddev)
        self.Noiser.apply_noise(self.images_clean)
        self.images_noisy = self.Noiser.images

        self.images_clean = torch.Tensor(self.images_clean).float()
        self.images_noisy = torch.Tensor(self.images_noisy).float()


    def __len__(self):
        return min(len(self.images_clean), len(self.images_noisy))

    def __getitem__(self, idx):
        return self.images_noisy[idx], self.images_clean[idx]