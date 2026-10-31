import matplotlib.pyplot as plt
from src.ImageLoader import ImageLoader
from src.ResolutionScaler import ResolutionScaler

def main():
    loader = ImageLoader(0, 0)
    loader.load_from_folder("./tests/test_images")
    loader.normalize()

    scaler_32 = ResolutionScaler(target_size=32)
    down_32 = scaler_32.downscale(loader.images)
    up_32 = scaler_32.upscale_bicubic()

    scaler_64 = ResolutionScaler(target_size=64)
    down_64 = scaler_64.downscale(loader.images)
    up_64 = scaler_64.upscale_bicubic()

    idx = 0
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    axes[0].imshow(loader.images[idx])
    axes[0].set_title("Original")

    axes[1].imshow(down_64[idx])
    axes[1].set_title("Downscale (64x64)")

    axes[2].imshow(up_64[idx])
    axes[2].set_title("Bicubic (64->256)")

    axes[3].imshow(down_32[idx])
    axes[3].set_title("Downscale (32x32)")

    axes[4].imshow(up_32[idx])
    axes[4].set_title("Bicubic (32->256)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
