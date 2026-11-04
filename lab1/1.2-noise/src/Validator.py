from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.restoration import denoise_bilateral
import torch
import numpy as np  
import matplotlib.pyplot as plt

class Validator:
    def __init__(self, model, lpips_net):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net,
            normalize=True
        ).to(self.device)

    def getMetrics(self, validationLoader, images_out = False):
        psnr_noaction = []
        psnr_model = []
        psnr_base = []

        ssim_noaction = []
        ssim_model = []
        ssim_base = []

        lpips_noaction =[]
        lpips_model = []
        lpips_base = []
        # breakpoint()

        for i, (clean_image, noisy_image) in enumerate(validationLoader):
            noisy_image, clean_image = noisy_image.to(self.device), clean_image.to(self.device)
            pred_model = self.model(noisy_image)
            img_model = noisy_image - pred_model
            pred_base = self.bilateral_denoise_batch(noisy_image)
            img_model = torch.clamp(img_model, 0, 1)

            print(f"validation on img {i+1}")

            psnr_noaction.append(peak_signal_noise_ratio(noisy_image, clean_image, data_range=1.0).item())
            psnr_model.append(peak_signal_noise_ratio(img_model, clean_image, data_range=1.0).item())
            psnr_base.append(peak_signal_noise_ratio(pred_base, clean_image, data_range=1.0).item())

            ssim_noaction.append(structural_similarity_index_measure(noisy_image, clean_image, data_range=1.0).item())
            ssim_model.append(structural_similarity_index_measure(img_model, clean_image, data_range=1.0).item())
            ssim_base.append(structural_similarity_index_measure(pred_base, clean_image, data_range=1.0).item())

            lpips_noaction.append(self.lpips(noisy_image, clean_image).item())
            lpips_model.append(self.lpips(img_model, clean_image).item())
            lpips_base.append(self.lpips(pred_base, clean_image).item())

            torch.cuda.empty_cache()

            if images_out and i % 20 == 0:
                image__np_cl = torch.squeeze(clean_image).permute(1, 2, 0).cpu().detach().numpy()
                image__np_noise = torch.squeeze(noisy_image).permute(1, 2, 0).cpu().detach().numpy()
                image__np_unet = torch.squeeze(img_model).permute(1, 2, 0).cpu().detach().numpy()
                image__np_base = torch.squeeze(pred_base).permute(1, 2, 0).cpu().detach().numpy()

                f, axarr = plt.subplots(2,2)
                axarr[0][0].axis('off')
                axarr[0][0].imshow(image__np_cl)
                axarr[0][0].title.set_text("clean image")

                axarr[0][1].axis('off')
                axarr[0][1].imshow(image__np_noise)
                axarr[0][1].title.set_text("noisy image")

                axarr[1][0].axis('off')
                axarr[1][0].imshow(image__np_base)
                axarr[1][0].title.set_text("bilateral denoising")

                axarr[1][1].axis('off')
                axarr[1][1].imshow(image__np_unet)
                axarr[1][1].title.set_text("unet denoising")
                plt.savefig(f"./out_images/comparison{i}.png")


        # breakpoint()
        return (
            [psnr_noaction, ssim_noaction, lpips_noaction],
            [psnr_base, ssim_base, lpips_base],
            [psnr_model, ssim_model, lpips_model]
        )

    def bilateral_denoise_batch(self, images, sigma_color=0.1, sigma_spatial=2):
        device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = images.dtype

        images_np = images.detach().permute(0, 2, 3, 1).cpu().numpy()
        denoised = np.stack([
            denoise_bilateral(
                img,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                channel_axis=-1,
            )
            for img in images_np
        ])
        return torch.from_numpy(
            denoised.transpose(0, 3, 1, 2).copy()
        ).to(device=device, dtype=dtype)
