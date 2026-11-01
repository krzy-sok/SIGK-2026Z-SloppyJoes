import skimage
import glob
import numpy as np

class ImageLoader:
    images: list[np.ndarray] = []
    x = 0
    y = 0

    def __init__(self, x, y ):
        self.images: list[np.ndarray] = []
        self.x = x
        self.y = y

    def load_from_folder(self, path="./images"):
        files =  glob.glob(path+"/*.png")
        for file in files:
            image: np.ndarray = skimage.io.imread(file)[self.y:self.y+256, self.x:self.x+256]
            self.images.append(image)

    def normalize(self):
        for i in range(len(self.images)):
            image = self.images[i]
            image2 = ( image / 255 )
            self.images[i] = image2