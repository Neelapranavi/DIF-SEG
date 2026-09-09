import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Compact U-Net for RGB medical-image lesion segmentation."""

    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        channels = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(channels, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))
            channels = feature

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)
