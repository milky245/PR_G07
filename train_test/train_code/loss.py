import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19(nn.Module):
    def __init__(self, vgg19_npy_path=None):
        super(Vgg19, self).__init__()
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')

    def build_conv4_4(self, rgb, include_fc=False):
        rgb_scaled = (rgb + 1) * 127.5
        blue, green, red = torch.chunk(rgb_scaled, 3, dim=1)
        bgr = torch.cat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], dim=1)
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = F.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = F.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = F.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = F.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = F.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = F.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = F.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = F.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = F.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = F.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = F.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = F.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        return self.conv4_4

    def max_pool(self, bottom, name):
        return F.max_pool2d(bottom, kernel_size=2, stride=2, padding=0)

    def conv_layer(self, bottom, name):
        filt = self.get_conv_filter(name)
        conv = F.conv2d(bottom, filt, stride=1, padding=1)
        bias = self.get_bias(name)
        return bias + conv

    def fc_layer(self, bottom, name):
        shape = bottom.shape
        dim = np.prod(shape[1:])
        x = bottom.view(-1, dim)
        weights = self.get_fc_weight(name)
        biases = self.get_bias(name)
        fc = torch.matmul(x, weights) + biases
        return fc

    def get_conv_filter(self, name):
        return torch.tensor(self.data_dict[name][0], dtype=torch.float32)

    def get_bias(self, name):
        return torch.tensor(self.data_dict[name][1], dtype=torch.float32)

    def get_fc_weight(self, name):
        return torch.tensor(self.data_dict[name][0], dtype=torch.float32)

def vggloss_4_4(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_conv4_4(image_a)
    vgg_b = vgg_model.build_conv4_4(image_b)
    VGG_loss = F.l1_loss(vgg_a, vgg_b)
    h, w, c = vgg_a.shape[1:]
    VGG_loss = torch.mean(VGG_loss) / (h * w * c)
    return VGG_loss

def wgan_loss(discriminator, real, fake, patch=True, channel=32, name='discriminator', lambda_=2):
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)

    d_loss_real = - torch.mean(real_logits)
    d_loss_fake = torch.mean(fake_logits)

    d_loss = d_loss_real + d_loss_fake
    g_loss = - d_loss_fake

    alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    differences = fake - real
    interpolates = real + (alpha * differences)
    inter_logit = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=inter_logit, inputs=interpolates, grad_outputs=torch.ones(inter_logit.size()).to(real.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1]))
    gradient_penalty = torch.mean((slopes - 1.) ** 2)
    d_loss += lambda_ * gradient_penalty

    return d_loss, g_loss

def gan_loss(discriminator, real, fake):
    real_logit = discriminator(real)
    fake_logit = discriminator(fake)

    real_logit = torch.sigmoid(real_logit)
    fake_logit = torch.sigmoid(fake_logit)

    g_loss = -torch.mean(torch.log(fake_logit))
    d_loss = -torch.mean(torch.log(real_logit) + torch.log(1. - fake_logit))

    return d_loss, g_loss

def lsgan_loss(discriminator, real, fake):
    real_logit = discriminator(real)
    fake_logit = discriminator(fake)

    g_loss = torch.mean((fake_logit - 1) ** 2)
    d_loss = 0.5 * (torch.mean((real_logit - 1) ** 2) + torch.mean(fake_logit ** 2))

    return d_loss, g_loss

def total_variation_loss(image, k_size=1):
    h, w = image.shape[2:]
    tv_h = torch.mean((image[:, :, k_size:, :] - image[:, :, :h - k_size, :]) ** 2)
    tv_w = torch.mean((image[:, :, :, k_size:] - image[:, :, :, :w - k_size]) ** 2)
    tv_loss = (tv_h + tv_w) / (3 * h * w)
    return tv_loss

def recon_loss(output, photo_batch, superpixel_batch):
    vgg = Vgg19('vgg19_no_fc.npy').cuda()
    vgg_output = vgg.build_conv4_4(output)
    vgg_photo = vgg.build_conv4_4(photo_batch)
    vgg_superpixel = vgg.build_conv4_4(superpixel_batch)
    h, w, c = vgg_output.shape[1:]
    photo_loss = torch.mean(F.l1_loss(vgg_photo, vgg_output)) / (h * w * c)
    superpixel_loss = torch.mean(F.l1_loss(vgg_superpixel, vgg_output)) / (h * w * c)
    return photo_loss + superpixel_loss

def generator_loss(output, cartoon_batch):
    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(cartoon_batch, cartoon_batch, r=5, eps=2e-1)
    gray_fake, gray_cartoon = color_shift(output, cartoon_batch)
    d_loss_gray, g_loss_gray = lsgan_loss(Discriminator(), gray_cartoon, gray_fake)
    d_loss_blur, g_loss_blur = lsgan_loss(Discriminator(), blur_cartoon, blur_fake)
    tv_loss = total_variation_loss(output)
    return 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss

def discriminator_loss(cartoon_batch, output):
    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(cartoon_batch, cartoon_batch, r=5, eps=2e-1)
    gray_fake, gray_cartoon = color_shift(output, cartoon_batch)
    d_loss_gray, _ = lsgan_loss(Discriminator(), gray_cartoon, gray_fake)
    d_loss_blur, _ = lsgan_loss(Discriminator(), blur_cartoon, blur_fake)
    return d_loss_blur + d_loss_gray

if __name__ == '__main__':
    pass
