import torch.nn as nn
import torch
from torchvision.models import vgg16, VGG16_Weights

class CharbonnierLoss(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, denoised, clean):
        root_contents = torch.square(denoised - clean) + self.eps**2
        return torch.mean( torch.sqrt(root_contents))


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=16):
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
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.criterion = nn.L1Loss()

    def train(self, mode=True):
        # Keep the frozen feature extractor in evaluation mode
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, pred, target):
        # pred and target: [B, 3, H, W], values in [0, 1]
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_features = self.features(pred)

        # Target features need no gradient
        with torch.no_grad():
            target_features = self.features(target)

        return self.criterion(pred_features, target_features)
