import torch
import torch.nn.functional as F
import numpy as np

def box_filter(x, r):
    ch = x.shape[1]
    weight = 1 / ((2 * r + 1) ** 2)
    box_kernel = torch.ones((ch, 1, 2 * r + 1, 2 * r + 1), dtype=torch.float32) * weight
    box_kernel = box_kernel.to(x.device)
    output = F.conv2d(x, box_kernel, padding=r, groups=ch)
    return output

def guided_filter(x, y, r, eps=1e-2):
    N = box_filter(torch.ones_like(x), r)
    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output

if __name__ == '__main__':
    pass
