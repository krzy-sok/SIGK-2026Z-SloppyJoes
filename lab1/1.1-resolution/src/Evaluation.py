import os
import warnings
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.ResolutionScaler import ResolutionScaler
from src.SuperResolutionModel import SuperResolutionUModel

class Evaluator:
    def __init__(self, device: torch.device | None = None, lpips_net: str = "alex"):
        self.device = (
            device
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        # lpips uses depraceted torchvision API (pretrained=True), so silence the warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
            self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False).to(self.device)
        self.lpips_fn.eval()

    @staticmethod
    def compute_sne(clean: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sum(np.square(clean - pred)))

    @staticmethod
    def compute_psnr(clean: np.ndarray, pred: np.ndarray) -> float:
        return float(peak_signal_noise_ratio(clean, pred, data_range=1.0))

    @staticmethod
    def compute_ssim(clean: np.ndarray, pred: np.ndarray) -> float:
        return float(
            structural_similarity(clean, pred, channel_axis=-1, data_range=1.0)
        )

    def compute_lpips(self, clean_t: torch.Tensor, pred_t: torch.Tensor) -> float:
        # expected tensors [B, 3, H, W] normalized to [-1, 1]
        with torch.no_grad():
            clean_norm = clean_t * 2.0 - 1.0
            pred_norm = pred_t * 2.0 - 1.0
            val = self.lpips_fn(clean_norm, pred_norm)
            return float(val.mean().item())

    def evaluate_resolution(
        self,
        validate_dir: str,
        model: SuperResolutionUModel,
        low_res_size: int,
        max_images: int | None = None,
    ) -> dict[str, dict[str, float]]:
        files = sorted(
            [
                os.path.join(validate_dir, f)
                for f in os.listdir(validate_dir)
                if os.path.isfile(os.path.join(validate_dir, f)) and f.endswith(".png")
            ]
        )
        if max_images is not None:
            files = files[:max_images]

        scaler = ResolutionScaler(target_size=low_res_size)
        model.to(self.device)
        model.eval()

        baseline_name = f"opencv_bicubic_resize_{low_res_size}_x_{low_res_size}"
        model_name = f"unet_super_resolution_{low_res_size}_x_{low_res_size}"

        metrics_acc = {
            baseline_name: {"SNE": [], "PSNR": [], "SSIM": [], "LPIPS": []},
            model_name: {"SNE": [], "PSNR": [], "SSIM": [], "LPIPS": []},
        }

        with torch.no_grad():
            for filepath in files:
                clean_pil = Image.open(filepath).convert("RGB").crop((0, 0, 256, 256))
                clean_np = np.asarray(clean_pil, dtype=np.float32) / 255.0

                # low res image
                low_res_np = np.clip(scaler.downscale_single(clean_np), 0.0, 1.0).astype(
                    np.float32
                )

                # baseline method: open cv resize, bicubic
                bicubic_np = scaler.upscale_bicubic_single(
                    low_res_np, output_size=256, max_val=1.0
                ).astype(np.float32)

                # u net model prediction
                low_res_t = (
                    torch.from_numpy(low_res_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(self.device)
                )
                pred_t = model(low_res_t)
                pred_np = (
                    pred_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                )

                clean_t = (
                    torch.from_numpy(clean_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(self.device)
                )
                bicubic_t = (
                    torch.from_numpy(bicubic_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(self.device)
                )

                # metrics for baseline method
                metrics_acc[baseline_name]["SNE"].append(
                    self.compute_sne(clean_np, bicubic_np)
                )
                metrics_acc[baseline_name]["PSNR"].append(
                    self.compute_psnr(clean_np, bicubic_np)
                )
                metrics_acc[baseline_name]["SSIM"].append(
                    self.compute_ssim(clean_np, bicubic_np)
                )
                metrics_acc[baseline_name]["LPIPS"].append(
                    self.compute_lpips(clean_t, bicubic_t)
                )

                # metrics for trainde model
                metrics_acc[model_name]["SNE"].append(
                    self.compute_sne(clean_np, pred_np)
                )
                metrics_acc[model_name]["PSNR"].append(
                    self.compute_psnr(clean_np, pred_np)
                )
                metrics_acc[model_name]["SSIM"].append(
                    self.compute_ssim(clean_np, pred_np)
                )
                metrics_acc[model_name]["LPIPS"].append(
                    self.compute_lpips(clean_t, pred_t)
                )

        summary: dict[str, dict[str, float]] = {}
        for method, m_dict in metrics_acc.items():
            summary[method] = {
                metric_name: float(np.mean(values))
                for metric_name, values in m_dict.items()
            }
        return summary

    @staticmethod
    def format_results_table(results: dict[str, dict[str, float]]) -> str:
        header = f"{'Method':<40} | {'SNE':>10} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8}"
        sep = "-" * len(header)
        lines = [header, sep]
        for method, vals in results.items():
            lines.append(
                f"{method:<40} | {vals['SNE']:>10.2f} | {vals['PSNR']:>8.4f} | {vals['SSIM']:>8.4f} | {vals['LPIPS']:>8.4f}"
            )
        return "\n".join(lines)

    def save_visual_comparison(
        self,
        validate_dir: str,
        model_32: SuperResolutionUModel,
        model_64: SuperResolutionUModel,
        output_path: str = "./image/results/comparison_results.png",
        image_idx: int = 0,
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        files = sorted(
            [
                os.path.join(validate_dir, f)
                for f in os.listdir(validate_dir)
                if os.path.isfile(os.path.join(validate_dir, f)) and f.endswith(".png")
            ]
        )
        clean_pil = Image.open(files[image_idx]).convert("RGB").crop((0, 0, 256, 256))
        clean_np = np.asarray(clean_pil, dtype=np.float32) / 255.0

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for row, (size, model) in enumerate([(64, model_64), (32, model_32)]):
            scaler = ResolutionScaler(target_size=size)
            low_res_np = np.clip(scaler.downscale_single(clean_np), 0.0, 1.0).astype(
                np.float32
            )
            bicubic_np = scaler.upscale_bicubic_single(
                low_res_np, output_size=256, max_val=1.0
            ).astype(np.float32)

            model.to(self.device)
            model.eval()
            with torch.no_grad():
                low_res_t = (
                    torch.from_numpy(low_res_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(self.device)
                )
                pred_np = (
                    model(low_res_t)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            bic_psnr = self.compute_psnr(clean_np, bicubic_np)
            bic_ssim = self.compute_ssim(clean_np, bicubic_np)
            mod_psnr = self.compute_psnr(clean_np, pred_np)
            mod_ssim = self.compute_ssim(clean_np, pred_np)

            axes[row, 0].imshow(clean_np)
            axes[row, 0].set_title("Original (256x256)")

            axes[row, 1].imshow(low_res_np)
            axes[row, 1].set_title(f"Input ({size}x{size})")

            axes[row, 2].imshow(bicubic_np)
            axes[row, 2].set_title(
                f"OpenCV Bicubic ({size}->256)\nPSNR: {bic_psnr:.2f} | SSIM: {bic_ssim:.3f}"
            )

            axes[row, 3].imshow(pred_np)
            axes[row, 3].set_title(
                f"Trained U-Net ({size}->256)\nPSNR: {mod_psnr:.2f} | SSIM: {mod_ssim:.3f}"
            )

            for col in range(4):
                axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Visual comparison saved to {output_path}")
