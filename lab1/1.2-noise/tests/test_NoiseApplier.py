import pytest
from src.ImageLoader import ImageLoader
from src.NoiseApplier import NoiseApplier
import numpy as np


def test_noise_count():
    loader = ImageLoader(0,0)
    loader.load_from_folder("./tests/test_images")
    noiseApp = NoiseApplier(0.03)
    noiseApp.apply_noise(loader.images)
    assert(len(noiseApp.images) == 10)
    assert(len(noiseApp.images) == len(loader.images))

def test_noise_not_original():
    loader = ImageLoader(0,0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()
    noiseApp = NoiseApplier(0.03)
    noiseApp.apply_noise(loader.images)
    for img_orig, img_noise in zip(loader.images, noiseApp.images):
        assert(img_noise.shape == (256,256,3))
        assert( not np.array_equal(img_orig, img_noise))


def test_noise_range():
    loader = ImageLoader(0,0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()
    noiseApp = NoiseApplier(0.03)
    noiseApp.apply_noise(loader.images)
    for image in  noiseApp.images:
        assert(np.min(image) >=0 and np.min(image) <= 1)
        assert(np.max(image) >=0 and np.max(image) <= 1)
        assert(image.max() >= image.min())
