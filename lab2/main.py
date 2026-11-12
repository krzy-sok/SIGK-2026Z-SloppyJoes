import argparse
import sys
import numpy as np
from tabulate import tabulate
import torch
from torch.utils.data import DataLoader
from src import ImageDownloader, NikonDataset
from src.NAFNet import NafNet
from src.Trainer import Trainer
from src.Validator import Validator
import matplotlib.pyplot as plt
import datetime

def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", action="store_true")
    parser.add_argument("--display_nikon_comp", action="store_true")
    parser.add_argument("--model", type=str, default="nafnet")
    parser.add_argument("--naf_down", type=list, default=[1,1,1,8])
    parser.add_argument("--naf_up", type=list, default=[1,1,1,1])
    parser.add_argument("--naf_mid", type=int, default=1)
    parser.add_argument("--out_name", default=f"{datetime.datetime.now().strftime("%d/%m/%Y_%H-%M-%S")}.torch")
    parser.add_argument("--lpips_net", default="vgg")

    args = parser.parse_args(arguments[1:])
    if args.d:
        ImageDownloader.download_raise_nef("./RAISE_258.csv","./images", 30)
        return

    if args.display_nikon_comp:
        image_dataset = NikonDataset.ImageDataset(count = 1)
        image_loader = DataLoader(image_dataset,batch_size=1)
        for img_clean, img_nikon in image_loader:
            f, axarr = plt.subplots(1,2)
            axarr[0].axis('off')
            axarr[0].imshow(img_clean[0].permute(1, 2, 0))
            axarr[0].title.set_text("clean image")

            axarr[1].axis('off')
            axarr[1].imshow(img_nikon[0].permute(1, 2, 0))
            axarr[1].title.set_text("noisy image")

            plt.show()
            return
    model = None

    match(args.model):
        case "nafnet":
            model = NafNet(
                down_blocks=args.naf_down,
                up_blocks=args.naf_up,
                middle_block_count=args.naf_mid
            )
        case _:
            print("no model chosen. Please run again with --model parameter. Possible values: nafnet")
            return

    train_datataset = NikonDataset.ImageDataset(count = 40)
    train_loader = DataLoader(train_datataset, batch_size=10)
    trainer = Trainer(model, learning_rate=0.001, epochs=50)
    losses = trainer.train_model(train_loader)
    torch.save(model.state_dict(), f"./{args.out_name}")

    plt.plot(losses)
    plt.savefig(f"losses_{args.out_name}.png")

    validator = Validator(model, args.lpips_net)
    validation_datatset = NikonDataset.ImageDataset(count = 20, offset=80)
    validatiopn_loader = DataLoader(validation_datatset, batch_size=1)
    noaction_metrics, model_metrics = validator.getMetrics(validatiopn_loader, images_out=True)

    noaction_metrics = ["no model"] + [np.mean(x) for x in noaction_metrics]
    model_metrics= [model.__class__.__name__] + [np.mean(x) for x in model_metrics]

    headers = ["model", "PSNR", "SSIM", f"LPIPS({args.lpips_net})"]
    table = tabulate([noaction_metrics, model_metrics], headers= headers)
    print(table)
    with open(f"./{args.out_name}.txt", "w", encoding="utf-8") as f:
        f.write(table)

if __name__ == "__main__":
    main(sys.argv)
