import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        root_contents = torch.square(pred - target) + self.eps**2
        return torch.mean(torch.sqrt(root_contents))

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer: int = 16):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights)

        self.features = nn.Sequential(
            *list(vgg.features.children())[:feature_layer]
        )

        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization for RGB inputs in [0, 1]
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self.criterion = nn.L1Loss()

    def train(self, mode: bool = True):
        # Keep the frozen feature extractor in evaluation mode
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred and target: [B, 3, H, W], values in [0, 1]
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_features = self.features(pred)

        # Target features dont need gradient
        with torch.no_grad():
            target_features = self.features(target)

        return self.criterion(pred_features, target_features)
