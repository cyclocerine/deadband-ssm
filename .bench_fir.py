import time

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
B, T, D = 1024, 30, 32
x = torch.randn(B, T, D)
k = torch.randn(T, D)


def bench(f, n=3):
    for _ in range(2):
        f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return 1000 * (time.perf_counter() - t0) / n


def dense():
    W = torch.zeros(D, D, T)
    idx = torch.arange(D)
    W[idx, idx, :] = k.t()
    return F.conv1d(x.transpose(1, 2), W, padding=T - 1)[..., :T].transpose(1, 2)


def depthwise():
    return F.conv1d(x.transpose(1, 2), k.flip(0).t().unsqueeze(1), padding=T - 1, groups=D)[..., :T].transpose(1, 2)


def pads():
    y = k[0].view(1, 1, -1) * x
    for j in range(1, T):
        y = y + F.pad(k[j].view(1, 1, -1) * x[:, :-j], (0, 0, j, 0))
    return y


def inplace():
    y = k[0].view(1, 1, -1) * x
    for j in range(1, T):
        y[:, j:] = y[:, j:] + k[j] * x[:, :-j]
    return y


print(f"dense-diag conv   {bench(dense):7.1f} ms")
print(f"depthwise conv    {bench(depthwise):7.1f} ms")
print(f"pad accumulate    {bench(pads):7.1f} ms")
print(f"inplace accumulate{bench(inplace):7.1f} ms")
print("max diff dense vs depthwise", float((dense() - depthwise()).abs().max()))
