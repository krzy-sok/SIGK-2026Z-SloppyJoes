from src.ImageDataset import ImageDataset
from src.Unet import Unet
from src.Validator import Validator
from torch.utils.data import DataLoader
import torch
from src.Training import Trainer
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument("--sigma",default=0.03, type=float)
    parser.add_argument("--load_model", type=str, default ="")
    parser.add_argument("-show_images", action="store_true")
    parser.add_argument("--lpips_net", default="vgg")
    parser.add_argument("--out_name", default="1.2-denoiser-model")
    args = parser.parse_args(arguments[1:])
    noise_sigma = args.sigma
    model = Unet()
    if(args.load_model != ""):
        model.load_state_dict(torch.load(args.load_model, weights_only=True))
        model.eval()
    else:
        dataset =  ImageDataset("../../images/train", noise_sigma)
        trainer = Trainer(model, 0.01, 50)
        dataLoader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers= 8)
        losses = trainer._train_model(dataLoader)
        torch.save(model.state_dict(), f"./{args.out_name}")
        plt.plot([x for x in range(len(losses))], losses)
        plt.savefig("./loss.png")

    print("validating model")
    validationDataset = ImageDataset("../../images/validate", noise_sigma)
    validator = Validator(model, args.lpips_net)
    validationLoader = DataLoader(validationDataset, batch_size=1, shuffle=True)

    noaction_metrics, base_metrics, model_metrics  = validator.getMetrics(validationLoader, args.show_images)
    # breakpoint()
    noaction_metrics = ["no model"] + [np.mean(x) for x in noaction_metrics]
    base_metrics= ["skimage.restoration.denoise_bilateral"] + [np.mean(x) for x in base_metrics]
    model_metrics= [model.__class__.__name__] + [np.mean(x) for x in model_metrics]

    headers = ["model", "PSNR", "SSIM", f"LPIPS({args.lpips_net})"]
    table = tabulate([noaction_metrics ,base_metrics, model_metrics], headers= headers)
    print(table)
    with open(f"./{args.out_name}.txt", "w", encoding="utf-8") as f:
        f.write(table)



if __name__ == "__main__":
    main(sys.argv)
