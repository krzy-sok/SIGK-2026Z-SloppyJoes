from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
import numpy as np  
import matplotlib.pyplot as plt
import torch.nn as nn

class Validator:
    def __init__(self, model: nn.Module, lpips_net):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net,
            normalize=True
        ).to(self.device)

    def getMetrics(self, validationLoader, images_out = False):
        psnr_noaction = []
        psnr_model = []

        ssim_noaction = []
        ssim_model = []

        lpips_noaction =[]
        lpips_model = []

        for i, (clean_image, noisy_image) in enumerate(validationLoader):
            noisy_image, clean_image = noisy_image.to(self.device), clean_image.to(self.device)
            pred_model = self.model(noisy_image)
            img_model = pred_model
            img_model = torch.clamp(img_model, 0, 1)

            print(f"validation on img {i+1}")

            psnr_noaction.append(peak_signal_noise_ratio(noisy_image, clean_image, data_range=1.0).item())
            psnr_model.append(peak_signal_noise_ratio(img_model, clean_image, data_range=1.0).item())

            ssim_noaction.append(structural_similarity_index_measure(noisy_image, clean_image, data_range=1.0).item())
            ssim_model.append(structural_similarity_index_measure(img_model, clean_image, data_range=1.0).item())

            lpips_noaction.append(self.lpips(noisy_image, clean_image).item())
            lpips_model.append(self.lpips(img_model, clean_image).item())

            torch.cuda.empty_cache()

            if images_out:
                image__np_cl = torch.squeeze(clean_image).permute(1, 2, 0).cpu().detach().numpy()
                image__np_noise = torch.squeeze(noisy_image).permute(1, 2, 0).cpu().detach().numpy()
                image__np_model = torch.squeeze(img_model).permute(1, 2, 0).cpu().detach().numpy()

                f, axarr = plt.subplots(3,1)
                axarr[0].axis('off')
                axarr[0].imshow(image__np_cl)
                axarr[0].title.set_text("clean image")

                axarr[1].axis('off')
                axarr[1].imshow(image__np_noise)
                axarr[1].title.set_text("nicon image")

                axarr[2].axis('off')
                axarr[2].imshow(image__np_model)
                axarr[2].title.set_text("NafNet")
                plt.savefig(f"./out_images/comparison{i}.png")


        # breakpoint()
        return (
            [psnr_noaction, ssim_noaction, lpips_noaction],
            [psnr_model, ssim_model, lpips_model]
        )