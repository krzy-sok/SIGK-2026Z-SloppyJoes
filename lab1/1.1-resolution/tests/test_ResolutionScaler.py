import pytest
import numpy as np
from src.ImageLoader import ImageLoader
from src.ResolutionScaler import ResolutionScaler

def test_scaler_count():
    loader = ImageLoader(0, 0)
    loader.load_from_folder("./tests/test_images")
    scaler = ResolutionScaler(target_size=32)
    scaler.downscale(loader.images)
    scaler.upscale_bicubic()
    assert len(scaler.images) == 10
    assert len(scaler.images) == len(loader.images)
    assert len(scaler.upscaled_images) == len(loader.images)

def test_downscale_shapes():
    loader = ImageLoader(0, 0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()

    scaler_32 = ResolutionScaler(target_size=32)
    scaler_32.downscale(loader.images)
    for image in scaler_32.images:
        assert image.shape == (32, 32, 3)

    scaler_64 = ResolutionScaler(target_size=64)
    scaler_64.downscale(loader.images)
    for image in scaler_64.images:
        assert image.shape == (64, 64, 3)

def test_upscale_not_original():
    loader = ImageLoader(0, 0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()

    for size in (32, 64):
        scaler = ResolutionScaler(target_size=size)
        scaler.downscale(loader.images)
        scaler.upscale_bicubic()
        for img_orig, img_up in zip(loader.images, scaler.upscaled_images):
            assert img_up.shape == (256, 256, 3)
            assert not np.array_equal(img_orig, img_up)

def test_scaler_range():
    loader = ImageLoader(0, 0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()

    for size in (32, 64):
        scaler = ResolutionScaler(target_size=size)
        scaler.downscale(loader.images)
        scaler.upscale_bicubic()
        for image in scaler.images + scaler.upscaled_images:
            assert np.min(image) >= 0 and np.min(image) <= 1
            assert np.max(image) >= 0 and np.max(image) <= 1
            assert image.max() >= image.min()
