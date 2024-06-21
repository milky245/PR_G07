import inspect
import os
import time
import numpy as np
import torch
import torch.nn as nn

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19(nn.Module):
    def __init__(self, vgg19_npy_path=None):
        super(Vgg19, self).__init__()
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')

    def build(self, rgb, include_fc=False):
        start_time = time.time()
        rgb_scaled = rgb * 127.5 + 1
        blue, green, red = torch.chunk(rgb_scaled, 3, dim=1)
        bgr = torch.cat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], dim=1)
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        if include_fc:
            self.fc6 = self.fc_layer(self.pool5, "fc6")
            self.relu6 = torch.nn.functional.relu(self.fc6)
            self.fc7 = self.fc_layer(self.relu6, "fc7")
            self.relu7 = torch.nn.functional.relu(self.fc7)
            self.fc8 = self.fc_layer(self.relu7, "fc8")
            self.prob = torch.nn.functional.softmax(self.fc8, dim=1)
            self.data_dict = None
        print("Finished building vgg19: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return torch.nn.functional.avg_pool2d(bottom, kernel_size=2, stride=2, padding=0)

    def max_pool(self, bottom, name):
        return torch.nn.functional.max_pool2d(bottom, kernel_size=2, stride=2, padding=0)

    def conv_layer(self, bottom, name):
        filt = self.get_conv_filter(name)
        conv = torch.nn.functional.conv2d(bottom, filt, stride=1, padding=1)
        bias = self.get_bias(name)
        return torch.nn.functional.relu(conv + bias)

    def fc_layer(self, bottom, name):
        shape = bottom.shape
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = bottom.view(-1, dim)
        weights = self.get_fc_weight(name)
        biases = self.get_bias(name)
        fc = torch.matmul(x, weights) + biases
        return fc

    def get_conv_filter(self, name):
        return torch.Tensor(self.data_dict[name][0])

    def get_bias(self, name):
        return torch.Tensor(self.data_dict[name][1])

    def get_fc_weight(self, name):
        return torch.Tensor(self.data_dict[name][0])
