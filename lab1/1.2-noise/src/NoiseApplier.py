import glob
import numpy as np
import skimage

class NoiseApplier():
    images: list[np.ndarray] =[]

    def __init__(self, noise_stddev, noise_mean = 0):
        self.stddev = noise_stddev
        self.mean = noise_mean

    def apply_noise(self, images_in: list[np.ndarray], max_val = 1):
        for image in images_in:
            noise = np.random.normal(self.mean, self.stddev, image.shape)
            image_noise = np.clip(image + noise, 0, max_val)
            self.images.append(image_noise)
