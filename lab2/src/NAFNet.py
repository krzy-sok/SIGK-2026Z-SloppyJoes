import torch.nn as nn
import torch
import torch.nn.functional as F
from .LayerNorm2d import LayerNorm2d

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

 
class NafBlock(nn.Module):
    def __init__(self, channels, DW_expand=2, FFN_expand = 2, drop_rate =0):
        super(NafBlock, self).__init__()
        self.DW_channels = channels* DW_expand
        self.FFN_channels = channels * FFN_expand
        self.drop_rate = drop_rate

        self.layerNorm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=self.DW_channels, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.DW_channels, out_channels=self.DW_channels, kernel_size=3, padding=1, stride=1, groups=self.DW_channels)
        self.gate = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels= self.DW_channels//2,
                out_channels=self.DW_channels//2,
                kernel_size= 1,
                padding= 0,
                stride= 1, groups=1, bias=True
            )
        )

        self.conv3 = nn.Conv2d(in_channels=self.DW_channels //2, out_channels=channels, kernel_size=1, stride=1)
        self.layerNorm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=self.FFN_channels, kernel_size=1, padding=0, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.FFN_channels //2, out_channels=channels, kernel_size=1, padding=0, stride=1)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        original = x
        x = self.layerNorm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = x * self.beta + original
        midpoint = x

        x = self.layerNorm2(x)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)

        return x * self.gamma + midpoint


class NafNet(nn.Module):
    def __init__(self, img_channels = 3, firstStep_channels = 16, down_blocks = [], up_blocks = [], middle_block_count = [] ):
        super().__init__()
        self.start = nn.Conv2d(in_channels=img_channels, out_channels=firstStep_channels, kernel_size=3, padding=1)
        self.finish = nn.Conv2d(in_channels=firstStep_channels, out_channels=img_channels, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.down = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.up = nn.ModuleList()
        self.middle = nn.ModuleList()

        current_channels = firstStep_channels
        for blocks_in_layer in down_blocks:
            self.encoders.append(
                nn.Sequential(
                    *[NafBlock(current_channels) for i in range(blocks_in_layer)]
                )
            )
            self.down.append(
                nn.Conv2d(in_channels=current_channels, out_channels=2*current_channels, kernel_size=2, stride=2)
            )
            current_channels *= 2

        self.middle = nn.Sequential(
            *[NafBlock(current_channels) for i in range(middle_block_count)]
        )

        for blocks_in_layer in up_blocks:
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=current_channels, out_channels=2*current_channels, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            current_channels = current_channels // 2

            self.decoders.append(
                nn.Sequential(\
                    *[NafBlock(current_channels) for i in range(blocks_in_layer)]
                )
            )
        self.padder_size = 2**(len(self.encoders))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.apply_padding(x)
        x = self.start(x)
        encoder_results = []

        for encoder, down in zip(self.encoders, self.down):
            x= encoder(x)
            encoder_results.append(x)
            x = down(x)
        x = self.middle(x)
        for decoder, up, skip in zip(self.decoders, self.up, encoder_results[::-1]):
            x =up(x)
            x = x+skip
            x = decoder(x)
        x = self.finish(x)
        return x[:,:,:H,:W]

            
    def apply_padding(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x