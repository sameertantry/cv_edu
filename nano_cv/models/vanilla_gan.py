import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.conv_dropout = nn.Dropout2d()

    def forward(self, x):
        x = self.conv_dropout(self.conv(x))
        x = F.relu(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv_block(x)
        y = self.pooling(x)

        return y


class UpBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=inner_channels)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=inner_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.upsample(x)
        x = self.conv(x)

        return x


class Generator(nn.Module):
    def __init__(self, img_size=64, latent_dim=7, channels=128):
        super().__init__()

        self.img_size = img_size
        self.channels = channels

        self.ll = nn.Linear(
            in_features=latent_dim,
            out_features=int(img_size / 4 * img_size / 4 * channels),
        )

        self.up1 = UpBlock(
            channels, 64, 32
        )  # 128 x IMG / 4 x IMG / 4  --> 32 x IMG / 2 x IMG / 2
        self.up2 = UpBlock(32, 16, 8)  # 32 x IMG / 2 x IMG / 2 --> 8 x IMG x IMG

        self.conv = nn.Conv2d(
            in_channels=8, out_channels=3, kernel_size=1
        )  # 8 x IMG x IMG --> 3 x IMG x IMG

    def forward(self, x):
        x = self.ll(x)
        x = x.view(-1, self.channels, int(self.img_size / 4), int(self.img_size / 4))

        x = self.up1(x)
        x = self.up2(x)

        x = self.conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.channels = channels

        self.down1 = DownBlock(3, 8)  # 64x64 --> 32x32
        self.down2 = DownBlock(8, 16)  # 32x32 --> 16x16
        self.down3 = DownBlock(16, 32)  # 16x16x16 --> 32x8x8
        self.down4 = DownBlock(32, channels)  # 32x8x8 --> 64x4x4

        self.fc1 = nn.Linear(in_features=channels * 4 * 4, out_features=channels)
        self.fc1_dropout = nn.Dropout()
        self.fc2 = nn.Linear(in_features=channels, out_features=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = x.view(-1, self.channels * 4 * 4)

        x = self.fc1_dropout(self.fc1(x))
        x = self.fc2(x)

        return x
