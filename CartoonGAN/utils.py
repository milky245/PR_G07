import torch
import numpy as np
import cv2
import os
from vgg19 import Vgg19

def leaky_relu(x, leak=0.2):
    return torch.maximum(x, leak * x)

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
            img = (image[k] + 1) * 127.5  # 将图像像素值转换回0-255范围
            img = img.transpose(1, 2, 0)  # 将形状从 [channels, height, width] 转换为 [height, width, channels]
            fused_image[i].append(img)
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    fused_image = fused_image.astype(np.uint8)  # 将图像转换为uint8类型
    cv2.imwrite(fused_dir, fused_image)

def print_fused_single_image(image, save_dir, name):
    fused_dir = os.path.join(save_dir, name)
    # 将图像像素值转换回0-255范围
    img = (image + 1) * 127.5
    img = img.transpose(1, 2, 0)  # 将形状从 [channels, height, width] 转换为 [height, width, channels]
    # 确保图像值在合法范围内
    img = np.clip(img, 0, 255)
    # 将图像转换为uint8类型
    img = img.astype(np.uint8)
    # 保存图像
    cv2.imwrite(fused_dir, img)


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
        image_crop = image[offset_h:offset_h + crop_size, offset_w:offset_w + crop_size]
        image_crop = image_crop / 127.5 - 1  # Normalize to [-1, 1]
        image_crop = np.transpose(image_crop, (2, 0, 1))  # Change shape to [channels, height, width]
        batch_data.append(image_crop)
    return np.asarray(batch_data)

def next_batch_no_resize(batch_size, image_size, filename_list):
    if len(filename_list) < batch_size:
        raise ValueError(f"Batch size {batch_size} is larger than the number of available images {len(filename_list)}.")
    idx = np.arange(0, len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        if image is None:
            raise ValueError(f"Image at {filename_list[idx[i]]} could not be read.")
        image = cv2.resize(image, (image_size, image_size))  # 将图像调整到指定大小
        if np.max(image) < 1.5:
            image = (image + 1) * 127.5
        image = image / 127.5 - 1  # 归一化到 [-1, 1]
        image = np.transpose(image, (2, 0, 1))  # 将形状从 [height, width, channels] 转换为 [channels, height, width]
        batch_data.append(image)
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
        image_crop = image[offset_h:offset_h + crop_size, offset_w:offset_w + crop_size]
        image_crop = image_crop / 127.5 - 1  # Normalize to [-1, 1]
        image_crop = np.transpose(image_crop, (2, 0, 1))  # Change shape to [channels, height, width]
        batch.append(image_crop)
        image_blur = cv2.GaussianBlur(image_crop.transpose(1, 2, 0), (5, 5), 0)  # Apply blur
        image_blur = image_blur / 127.5 - 1  # Normalize to [-1, 1]
        image_blur = np.transpose(image_blur, (2, 0, 1))  # Change shape to [channels, height, width]
        blur_batch.append(image_blur)
    return np.asarray(batch), np.asarray(blur_batch)

def vgg_loss(image_a, image_b):
    vgg_a, vgg_b = Vgg19('vgg19.npy'), Vgg19('vgg19.npy')
    vgg_a.build(image_a)
    vgg_b.build(image_b)
    VGG_loss = torch.mean(torch.abs(vgg_a.conv4_4 - vgg_b.conv4_4))
    h, w, c = vgg_a.conv4_4.shape[1:]
    VGG_loss = VGG_loss / (h * w * c)
    return VGG_loss

def l2_norm(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1):
    w_shape = list(w.shape)
    w = w.view(-1, w_shape[-1])
    u = torch.randn(1, w_shape[-1], requires_grad=False)

    for _ in range(iteration):
        v = l2_norm(torch.matmul(u, w.t()))
        u = l2_norm(torch.matmul(v, w))

    sigma = torch.matmul(v, torch.matmul(w, u.t()))
    w_norm = w / sigma
    return w_norm.view(*w_shape)

def conv_sn(x, channels, k_size, stride=1, name='conv2d'):
    w = torch.nn.Parameter(torch.Tensor(channels, x.shape[1], k_size, k_size))
    b = torch.nn.Parameter(torch.zeros(channels))
    output = torch.nn.functional.conv2d(x, spectral_norm(w), b, stride, padding=k_size // 2)
    return output
