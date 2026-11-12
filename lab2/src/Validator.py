import os
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms.functional as TF
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

    def histogram_equalization(self, img: torch.Tensor) -> torch.Tensor:
        img_uint8 = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        eq_uint8 = TF.equalize(img_uint8)
        return eq_uint8.float() / 255.0

    def getMetrics(self, validationLoader, images_out = False):
        self.model.eval()

        psnr_noaction = []
        psnr_baseline = []
        psnr_model = []

        ssim_noaction = []
        ssim_baseline = []
        ssim_model = []

        lpips_noaction = []
        lpips_baseline = []
        lpips_model = []

        if images_out:
            os.makedirs("./out_images", exist_ok=True)

        with torch.no_grad():
            for i, (clean_image, noisy_image) in enumerate(validationLoader):
                noisy_image, clean_image = noisy_image.to(self.device), clean_image.to(self.device)

                img_baseline = self.histogram_equalization(noisy_image)

                pred_model = self.model(noisy_image)
                img_model = noisy_image + pred_model
                # img_model = pred_model
                img_model = img_model.clamp(0, 1)
                # breakpoint()
                print(f"validation on img {i+1}")

                psnr_noaction.append(peak_signal_noise_ratio(noisy_image, clean_image, data_range=1.0).item())
                psnr_baseline.append(peak_signal_noise_ratio(img_baseline, clean_image, data_range=1.0).item())
                psnr_model.append(peak_signal_noise_ratio(img_model, clean_image, data_range=1.0).item())

                ssim_noaction.append(structural_similarity_index_measure(noisy_image, clean_image, data_range=1.0).item())
                ssim_baseline.append(structural_similarity_index_measure(img_baseline, clean_image, data_range=1.0).item())
                ssim_model.append(structural_similarity_index_measure(img_model, clean_image, data_range=1.0).item())

                lpips_noaction.append(self.lpips(noisy_image, clean_image).item())
                lpips_baseline.append(self.lpips(img_baseline, clean_image).item())
                lpips_model.append(self.lpips(img_model, clean_image).item())

                if images_out:
                    image_np_cl = torch.squeeze(clean_image).permute(1, 2, 0).cpu().detach().numpy()
                    image_np_noise = torch.squeeze(noisy_image).permute(1, 2, 0).cpu().detach().numpy()
                    image_np_base = torch.squeeze(img_baseline).permute(1, 2, 0).cpu().detach().numpy()
                    image_np_model = torch.squeeze(img_model).permute(1, 2, 0).cpu().detach().numpy()

                    f, axarr = plt.subplots(4, 1, figsize=(6, 12))
                    axarr[0].axis('off')
                    axarr[0].imshow(image_np_cl)
                    axarr[0].title.set_text("clean image (GT)")

                    axarr[1].axis('off')
                    axarr[1].imshow(image_np_noise)
                    axarr[1].title.set_text("nikon image (input)")

                    axarr[2].axis('off')
                    axarr[2].imshow(image_np_base)
                    axarr[2].title.set_text("baseline (hist eq)")

                    axarr[3].axis('off')
                    axarr[3].imshow(image_np_model)
                    axarr[3].title.set_text("NafNet")
                    plt.tight_layout()
                    plt.savefig(f"./out_images/comparison{i}.png")
                    plt.close(f)

        # breakpoint()
        return (
            [psnr_noaction, ssim_noaction, lpips_noaction],
            [psnr_baseline, ssim_baseline, lpips_baseline],
            [psnr_model, ssim_model, lpips_model]
        )