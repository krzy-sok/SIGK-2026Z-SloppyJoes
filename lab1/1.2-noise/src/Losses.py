import torch.nn as nn
import torch
from torchvision.models import vgg16

class CharbonnierLoss(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, denoised, clean):
        root_contents = torch.square(denoised - clean) + self.eps**2
        return torch.mean( torch.sqrt(root_contents))


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vgg = vgg16(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg_layers = vgg.to(self.device)
        self.feature_layers = feature_layers
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        loss = 0.0
        x_feats, y_feats = x, y
        for i, layer in enumerate(self.vgg_layers):
            x_feats = layer(x_feats)
            y_feats = layer(y_feats)
            if i in self.feature_layers:
                loss += self.criterion(x_feats, y_feats)
            if i == max(self.feature_layers):
                break
        return loss


class FFDNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, denoised, clean):
        # MSE averaged over all pixels and channels
        return 0.5 * torch.mean((denoised - clean) ** 2)