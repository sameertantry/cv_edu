import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, n_channels: int):
        super().__init__()

        self.size = img_size // 32
        self.n_channels = n_channels

        self.proj = nn.Linear(
            in_features=latent_dim, out_features=self.size * self.size * self.n_channels
        )

        self.convs = nn.Sequential(
            nn.BatchNorm2d(self.n_channels),
            UpBlock(self.n_channels, self.n_channels // 2),
            UpBlock(self.n_channels // 2, self.n_channels // 4),
            UpBlock(self.n_channels // 4, self.n_channels // 8),
            UpBlock(self.n_channels // 8, self.n_channels // 16),
            UpBlock(self.n_channels // 16, self.n_channels // 32),
            nn.Conv2d(self.n_channels // 32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = x.view(-1, self.n_channels, self.size, self.size)

        return self.convs(x)


class Discriminator(nn.Module):
    def __init__(self, img_size: int, n_channels: int):
        super().__init__()

        self.down_blocks = nn.Sequential(
            DownBlock(3, n_channels // 32),
            DownBlock(n_channels // 32, n_channels // 16),
            DownBlock(n_channels // 16, n_channels // 8),
            DownBlock(n_channels // 8, n_channels // 4),
            DownBlock(n_channels // 4, n_channels // 2),
            nn.Conv2d(n_channels // 2, n_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(n_channels, 1, kernel_size=1),
        )

        self.pred = nn.Sequential(
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.down_blocks(x)

        return self.pred(x)
