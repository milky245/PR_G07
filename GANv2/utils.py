import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from vgg19 import Vgg19

def leaky_relu(x, leak=0.2):
    return torch.maximum(x, torch.tensor(leak) * x)

def print_image(image, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fused_dir = os.path.join(save_dir, 'fused_image.jpg')
    fused_image = [0] * 8
    for i in range(8):
        fused_image[i] = []
        for j in range(8):
            k = i * 8 + j
            image[k] = (image[k] + 1) * 127.5
            fused_image[i].append(image[k])
            img_dir = os.path.join(save_dir, name + str(k) + '.jpg')
            cv2.imwrite(img_dir, image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image)

def print_fused_image(image, save_dir, name, n):
    fused_dir = os.path.join(save_dir, name)
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            image[k] = (image[k] + 1) * 127.5
            fused_image[i].append(image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image)

def get_filename_list(load_dir):
    filename_list = []
    for name in os.listdir(load_dir):
        file_name = os.path.join(load_dir, name)
        filename_list.append(file_name)
    return filename_list

def next_batch(batch_size, crop_size, filename_list):
    idx = np.arange(0, len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        img_h, img_w = np.shape(image)[:2]
        if np.max(image) < 1.5:
            image = (image + 1) * 127.5
        offset_h = np.random.randint(0, img_h - crop_size)
        offset_w = np.random.randint(0, img_w - crop_size)
        image_crop = image[offset_h:offset_h + crop_size,
                           offset_w:offset_w + crop_size]
        batch_data.append(image_crop / 127.5 - 1)
    return np.asarray(batch_data)

def next_blur_batch(batch_size, crop_size, filename_list):
    idx = np.arange(0, len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch, blur_batch = [], []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        img_h, img_w = np.shape(image)[:2]
        offset_h = np.random.randint(0, img_h - crop_size)
        offset_w = np.random.randint(0, img_w - crop_size)
        image_crop = image[offset_h:offset_h + crop_size,
                           offset_w:offset_w + crop_size]
        batch.append(image_crop / 127.5 - 1)
        image_blur = cv2.GaussianBlur(image_crop, (5, 5), 0)
        blur_batch.append(image_blur / 127.5 - 1)
    return np.asarray(batch), np.asarray(blur_batch)

def vgg_loss(image_a, image_b):
    # vgg_a, vgg_b = Vgg19('vgg19.npy'), Vgg19('vgg19.npy')
    # vgg_a.build(image_a)
    # vgg_b.build(image_b)
    # VGG_loss = F.l1_loss(vgg_a.conv4_4, vgg_b.conv4_4)
    # h, w, c = vgg_a.conv4_4.shape[1:]
    # VGG_loss = VGG_loss / (h * w * c)
    # return VGG_loss
    vgg_a, vgg_b = Vgg19('vgg19.npy').to(image_a.device), Vgg19('vgg19.npy').to(image_b.device)
    vgg_a_features = vgg_a(image_a)
    vgg_b_features = vgg_b(image_b)
    VGG_loss = F.l1_loss(vgg_a_features, vgg_b_features)
    return VGG_loss


def l2_norm(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        weight = getattr(self.module, self.name + "_bar")

        height = weight.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2_norm(torch.matmul(weight.data.view(height, -1).t(), u.data))
            u.data = l2_norm(torch.matmul(weight.data.view(height, -1), v.data))

        sigma = u.dot(weight.view(height, -1).mv(v))
        weight.data /= sigma

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            weight = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        weight = getattr(self.module, self.name)

        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]

        u = nn.Parameter(weight.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(weight.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2_norm(u.data)
        v.data = l2_norm(v.data)
        weight_bar = nn.Parameter(weight.data)

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", weight_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def conv_sn(x, channels, k_size, stride=1):
    conv = nn.Conv2d(in_channels=x.size(1), out_channels=channels, kernel_size=k_size, stride=stride, padding=k_size // 2)
    return SpectralNorm(conv)(x)
