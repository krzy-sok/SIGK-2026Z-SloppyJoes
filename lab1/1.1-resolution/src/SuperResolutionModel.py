import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperResolutionUModel(nn.Module):
    def __init__(self, output_size: int = 256, use_residual: bool = True):
        super(SuperResolutionUModel, self).__init__()
        self.output_size = output_size
        self.use_residual = use_residual

        # Encoder (wspólny rdzeń U-Net z zadaniem 2.2)
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.dec1 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(256, 64)
        self.dec3 = self.upconv_block(128, 32)
        self.dec4 = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Jeśli wejście jest mniejsze niż docelowe (np. 32x32 lub 64x64),
        # warstwa wejściowa modelu skaluje tensor przestrzenny do (output_size, output_size)
        if x.shape[-2:] != (self.output_size, self.output_size):
            x_up = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bicubic",
                align_corners=False,
            )
        else:
            x_up = x

        # Encoder
        e1 = self.enc1(x_up)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder + Skip connections
        d1 = self.dec1(e4)
        d1 = torch.cat([e3, d1], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([e2, d2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([e1, d3], dim=1)
        d4 = self.dec4(d3)

        if self.use_residual:
            return torch.clamp(x_up + d4, 0.0, 1.0)
        return torch.sigmoid(d4)
