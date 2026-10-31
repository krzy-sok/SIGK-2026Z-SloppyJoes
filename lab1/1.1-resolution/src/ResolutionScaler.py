import cv2
import numpy as np


class ResolutionScaler:
    images: list[np.ndarray] = []
    upscaled_images: list[np.ndarray] = []

    def __init__(self, target_size: int = 32, downscale_interpolation: int = cv2.INTER_AREA):
        self.target_size = target_size
        self.downscale_interpolation = downscale_interpolation
        self.images: list[np.ndarray] = []
        self.upscaled_images: list[np.ndarray] = []

    def downscale_single(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=self.downscale_interpolation,
        )

    def upscale_bicubic_single(
        self, image: np.ndarray, output_size: int = 256, max_val: float = 1.0
    ) -> np.ndarray:
        upscaled = cv2.resize(
            image,
            (output_size, output_size),
            interpolation=cv2.INTER_CUBIC,
        )
        return np.clip(upscaled, 0.0, max_val)

    def downscale(self, images_in: list[np.ndarray]) -> list[np.ndarray]:
        self.images = []
        for image in images_in:
            resized = self.downscale_single(image)
            self.images.append(resized)
        return self.images

    def upscale_bicubic(
        self,
        images_in: list[np.ndarray] | None = None,
        output_size: int = 256,
        max_val: float = 1.0,
    ) -> list[np.ndarray]:
        source_images = self.images if images_in is None else images_in
        self.upscaled_images = []
        for image in source_images:
            upscaled = self.upscale_bicubic_single(image, output_size=output_size, max_val=max_val)
            self.upscaled_images.append(upscaled)
        return self.upscaled_images
