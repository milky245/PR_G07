import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SpectralNorm, leaky_relu

def conv_sn(in_channels, out_channels, kernel_size, stride=1, padding=0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    return SpectralNorm(conv)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + shortcut

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
        self.conv1 = conv_sn(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_sn(32, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv_sn(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv_sn(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = conv_sn(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_out = conv_sn(512, 1, kernel_size=1, stride=1)

    def forward(self, x):
        patches = []
        batch_size = x.size(0)
        for i in range(4):
            idx = torch.randperm(batch_size)
            patch = x[idx][:, :, :self.patch_size, :self.patch_size]
            patch = F.relu(self.conv1(patch))
            patch = F.relu(self.conv2(patch))
            patches.append(patch)
        x = torch.cat(patches, dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv_out(x)
        x = torch.mean(x, dim=[2, 3])
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self, patch_size, use_bn=True):
        super(PatchDiscriminator, self).__init__()
        self.patch_size = patch_size
        self.use_bn = use_bn
        self.conv1 = conv_sn(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_sn(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv_sn(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv_sn(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = conv_sn(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = conv_sn(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv_out = conv_sn(128, 1, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x.size(0)
        idx = torch.randperm(batch_size)
        patch = x[idx][:, :, :self.patch_size, :self.patch_size]
        x = F.relu(self.conv1(patch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv_out(x)
        x = torch.mean(x, dim=[2, 3])
        return x
