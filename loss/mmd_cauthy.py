import torch
import numpy as np


def Cauchy_kernel(source, target, kernel_mul=10.0, kernel_num=1, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [1 / (L2_distance / bandwidth_temp + 1) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_cauthy_noaccelerate(source, target, kernel_mul=10.0, kernel_num=5, fix_sigma=None):  # None
    batch_size_sr = int(source.size()[0])
    batch_size_tar = int(target.size()[0])
    kernels = Cauchy_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size_sr, :batch_size_sr]
    YY = kernels[batch_size_sr:, batch_size_sr:]
    XY = kernels[:batch_size_sr, batch_size_sr:]
    YX = kernels[batch_size_sr:, :batch_size_sr]
    # loss = torch.mean(XX + YY - XY -YX)
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss
