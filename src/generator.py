import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_stack = nn.ModuleList([
            self.downsample(3, 64, apply_batchnorm=False),
            self.downsample(64, 128),
            self.downsample(128, 256),
            self.downsample(256, 512),
            self.downsample(512, 512),
            self.downsample(512, 512),
            self.downsample(512, 512),
            self.downsample(512, 512, apply_batchnorm=False),
        ])

        self.up_stack = nn.ModuleList([
            self.upsample(512, 512, apply_dropout=True),
            self.upsample(1024, 512, apply_dropout=True),
            self.upsample(1024, 512, apply_dropout=True),
            self.upsample(1024, 512),
            self.upsample(1024, 256),
            self.upsample(512, 128),
            self.upsample(256, 64),
        ])

        self.last = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        return torch.tanh(self.last(x))

    def downsample(self, in_channels, out_channels, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def upsample(self, in_channels, out_channels, apply_dropout=False):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels)]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)