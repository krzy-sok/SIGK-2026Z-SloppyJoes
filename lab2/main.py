import argparse
import sys
from torch.utils.data import DataLoader
from src import ImageDownloader, NikonDataset
import matplotlib.pyplot as plt

def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", action="store_true")
    parser.add_argument("--display_nikon_comp", action="store_true")
    args = parser.parse_args(arguments[1:])
    if args.d:
        ImageDownloader.download_raise_nef("./RAISE_258.csv","./images", 100)
        return

    if args.display_nikon_comp:
        image_dataset = NikonDataset.ImageDataset(count = 1)
        image_loader = DataLoader(image_dataset,batch_size=1)
        for img_clean, img_nikon in image_loader:
            f, axarr = plt.subplots(1,2)
            axarr[0].axis('off')
            axarr[0].imshow(img_clean[0])
            axarr[0].title.set_text("clean image")

            axarr[1].axis('off')
            axarr[1].imshow(img_nikon[0])
            axarr[1].title.set_text("noisy image")

            plt.show()


if __name__ == "__main__":
    main(sys.argv)
