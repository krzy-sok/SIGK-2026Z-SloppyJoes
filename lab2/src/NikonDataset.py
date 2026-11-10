from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import rawpy
import os

class ImageDataset(Dataset):
    def __init__(self, image_path: str = "./images", offset = 0, count = 100):
        self.images = sorted(
            [os.path.join(image_path, f) for f
             in os.listdir(image_path)
             if os.path.isfile(os.path.join(image_path, f))  and ".NEF" in f]
             )

        limit = offset+count
        if limit > len(self.images):
            limit = len(self.images)
        self.images = self.images[offset: limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path  = self.images[idx]
        with rawpy.imread(path) as raw:
            clean_img = self.get_linear_image(raw)
            nikon_img = self.nikon_Transform(clean_img)
            print("transforms done")
            transform = transforms.ToTensor()
            clean_transformed = transform(clean_img)
            nikon_transformed = transform(nikon_img)
            return clean_img, nikon_img

    def nikon_Transform(self: np.ndarray, image):
        image = self.reduce_exposure(image)
        print("exposure done")
        image = self.add_physical_noise(image)
        print("noise done")
        image = self.apply_gamma_correction(image)
        print("gamma done")
        return self.convert_to_8bit(image)


    def reduce_exposure(self, image: np.ndarray, ev_shift =-3):
        """
        Calculate the scale factor based on 'ev_shift' and multiply the image by it.
        """
        ev_linear = np.pow(2.0, ev_shift)
        return np.clip(image * ev_linear, 0, 1)


    def add_physical_noise(self, linear_img: np.ndarray, alpha=0.001, sigma=0.005):
        """
        Simulates physical camera sensor noise.
        alpha: controls Poisson (photon) noise
        sigma: controls Gaussian (electronic read) noise
        """
        noisy_img = np.random.poisson(linear_img / alpha) * alpha
        noisy_img += np.random.normal(0, sigma, linear_img.shape)
        return np.clip(noisy_img, 0.0, 1.0)


    def apply_gamma_correction(self, image: np.ndarray, gamma=2.2):
        im_gamma = 1/gamma
        return np.clip(np.pow(image, im_gamma), 0, 1)


    def convert_to_8bit(self, image: np.ndarray):
        return (image * 255).astype(np.uint8)


    def get_linear_image(self, raw):
        """
        Extracts the linear image from RAW data.
        Disables auto-brightness and gamma correction.
        Returns a float32 image in the range [0.0, 1.0].
        """
        linear_16bit = raw.postprocess(
            use_camera_wb=True,
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            half_size=True
        )
        return linear_16bit.astype(np.float32) / 65535.0