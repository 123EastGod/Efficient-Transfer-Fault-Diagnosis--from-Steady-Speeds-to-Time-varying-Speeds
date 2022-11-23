import torch


def Polynomial_kernel(source, target, a=1.0, c=0.5, d=5):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    xy = torch.mm(total, torch.transpose(total, 0, 1))
    # print(xy)
    # a /= torch.abs(torch.mean(xy.data))/(n_samples**2-n_samples)  # / n_samples * 2
    # (n_samples**2-n_samples) # / n_samples
    loss = (a*xy + c)**d
    # print(loss.size())   # .size()
    return loss


def mmd_polynomial_noaccelerate(source, target):  # None
    batch_size_sr = int(source.size()[0])
    batch_size_tar = int(target.size()[0])

    kernels = Polynomial_kernel(source, target, a=0.00001, c=0.5, d=3)
    # kernels = sigmoid_kernel(source, target, a=2.0, c=5)
    # print(kernels.size())
    XX = kernels[:batch_size_sr, :batch_size_sr]
    YY = kernels[batch_size_sr:, batch_size_sr:]
    XY = kernels[:batch_size_sr, batch_size_sr:]
    YX = kernels[batch_size_sr:, :batch_size_sr]
    # loss = torch.mean(XX + YY - XY -YX)
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss
