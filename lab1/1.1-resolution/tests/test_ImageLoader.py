import pytest
from src.ImageLoader import ImageLoader
import numpy as np

def test_load_image_size():
    loader = ImageLoader(0,0)
    loader.load_from_folder("./test_images")
    for image in loader.images:
        assert(image.shape == (256,256,3))

def test_load_count():
    loader = ImageLoader(0,0)
    loader.load_from_folder("./tests/test_images")
    assert(len(loader.images) == 10)

def test_normalized_vaues():
    loader = ImageLoader(0,0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()
    for image in loader.images:
        assert(image.shape == (256,256,3))
        assert(np.min(image) >=0 and np.min(image) <= 1)
        assert(np.max(image) >=0 and np.max(image) <= 1)
        assert(image.max() >= image.min())
