from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure, learned_perceptual_image_patch_similarity
from skimage.restoration import denoise_bilateral
import torch
import numpy as np

class Validator:
    def __init__(self, model):
        self.model = model

    def getMetrics(self, validationLoader):
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
            pred_model = self.model(noisy_image)
            pred_base = self.bilateral_denoise_batch(noisy_image)
            # breakpoint()
            # if(i&1):
            #     continue
            print(f"validation on img {i+1}")

            psnr_noaction.append(peak_signal_noise_ratio(noisy_image, clean_image, data_range=1.0).item())
            psnr_model.append(peak_signal_noise_ratio(pred_model, clean_image, data_range=1.0).item())
            psnr_base.append(peak_signal_noise_ratio(pred_base, clean_image, data_range=1.0).item())
            # print("psnr done")

            ssim_noaction.append(structural_similarity_index_measure(noisy_image, clean_image, data_range=1.0).item())
            ssim_model.append(structural_similarity_index_measure(pred_model, clean_image, data_range=1.0).item())
            ssim_base.append(structural_similarity_index_measure(pred_base, clean_image, data_range=1.0).item())
            # print("ssim done")

            # if(i==78):
            #     breakpoint()
            lpips_noaction.append(learned_perceptual_image_patch_similarity(noisy_image, clean_image).item())
            # print("lpips noacition done")
            lpips_model.append(learned_perceptual_image_patch_similarity(pred_model, clean_image).item())
            # print("lpips model done")

            lpips_base.append(learned_perceptual_image_patch_similarity(pred_base, clean_image).item())
            # print("lpips done")
            torch.cuda.empty_cache()

            if i >=80:
                # out of gpu mem
                break


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
