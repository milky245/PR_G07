import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(4)])
        self.deconv1_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv2_1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(x)))))
        x = self.res_blocks(x)
        x = F.relu(self.bn4(self.deconv1_2(F.relu(self.deconv1_1(x)))))
        x = F.relu(self.bn5(self.deconv2_2(F.relu(self.deconv2_1(x)))))
        x = self.conv_out(x)
        return x

class MultiPatchDiscriminator(nn.Module):
    def __init__(self, patch_size, use_bn=True):
        super(MultiPatchDiscriminator, self).__init__()
        self.patch_size = patch_size
        self.use_bn = use_bn
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        for i in range(4):
            conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(conv)
            if use_bn:
                self.conv_layers.append(nn.BatchNorm2d(32))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            in_channels = 32  # 更新输入通道数
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        if use_bn:
            self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.bn4 = nn.BatchNorm2d(512)
        self.conv_out = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        patches = []
        for i in range(4):
            batch_size = x.size(0)
            idx = torch.randint(0, x.size(2) - self.patch_size, (1,)).item()
            idy = torch.randint(0, x.size(3) - self.patch_size, (1,)).item()
            patch = x[:, :, idx:idx + self.patch_size, idy:idy + self.patch_size]
            for layer in self.conv_layers:
                patch = layer(patch)
            patches.append(patch)
        x = torch.cat(patches, dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)) if self.use_bn else self.conv3(x), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)) if self.use_bn else self.conv4(x), 0.2)
        x = torch.mean(self.conv_out(x), dim=[2, 3])
        return x
