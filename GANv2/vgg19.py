import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19(nn.Module):
    def __init__(self, vgg19_npy_path=None):
        super(Vgg19, self).__init__()
        if vgg19_npy_path is None:
            vgg19_npy_path = "vgg19.npy"
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        self.build()

    def build(self):
        self.conv1_1 = self._make_conv_layer("conv1_1")
        self.conv1_2 = self._make_conv_layer("conv1_2")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = self._make_conv_layer("conv2_1")
        self.conv2_2 = self._make_conv_layer("conv2_2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = self._make_conv_layer("conv3_1")
        self.conv3_2 = self._make_conv_layer("conv3_2")
        self.conv3_3 = self._make_conv_layer("conv3_3")
        self.conv3_4 = self._make_conv_layer("conv3_4")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = self._make_conv_layer("conv4_1")
        self.conv4_2 = self._make_conv_layer("conv4_2")
        self.conv4_3 = self._make_conv_layer("conv4_3")
        self.conv4_4 = self._make_conv_layer("conv4_4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = self._make_conv_layer("conv5_1")
        self.conv5_2 = self._make_conv_layer("conv5_2")
        self.conv5_3 = self._make_conv_layer("conv5_3")
        self.conv5_4 = self._make_conv_layer("conv5_4")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def _make_conv_layer(self, name):
        filt = torch.tensor(self.data_dict[name][0], dtype=torch.float32).permute(3, 2, 0, 1)
        bias = torch.tensor(self.data_dict[name][1], dtype=torch.float32)
        conv_layer = nn.Conv2d(in_channels=filt.shape[1],
                               out_channels=filt.shape[0],
                               kernel_size=filt.shape[2],
                               padding=filt.shape[2] // 2)
        conv_layer.weight = nn.Parameter(filt)
        conv_layer.bias = nn.Parameter(bias)
        return conv_layer

    def forward(self, x):
        h = self.pool1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        h = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(h)))))
        h = self.pool3(F.relu(self.conv3_4(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h)))))))))
        h = self.pool4(F.relu(self.conv4_4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h)))))))))
        h = self.pool5(F.relu(self.conv5_4(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(h)))))))))
        return h
