import torch
from torch import nn
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = self.downsample(4, 64, apply_batchnorm=False)
        self.down2 = self.downsample(64, 128)
        self.down3 = self.downsample(128, 256)
        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(512)
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, inp, target):
        x = torch.cat([inp, target], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = F.leaky_relu(self.batchnorm(self.conv(x)), 0.2)
        return self.last(x)

    def downsample(self, in_channels, out_channels, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)