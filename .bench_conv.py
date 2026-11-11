import time

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
B, T, D = 1024, 30, 32
x = torch.randn(B, T, D)
k3 = torch.randn(3, D)
k30 = torch.randn(30, D)


def bench(f, n=5):
    for _ in range(2):
        f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return 1000 * (time.perf_counter() - t0) / n


def taps(k):
    def f():
        y = k[0] * x
        for j in range(1, k.shape[0]):
            y[:, j:] = y[:, j:] + k[j] * x[:, :-j]
        return y

    return f


def conv(k):
    w = k.flip(0).t().unsqueeze(1)
    return lambda: F.conv1d(x.transpose(1, 2), w, padding=k.shape[0] - 1, groups=D)[:, :, :T].transpose(1, 2)


from core.layers import LinearSSM

ssm = LinearSSM(D, d_state=8, form="loop")
print(f"conv1d k3        {bench(conv(k3)):7.1f} ms")
print(f"conv1d k30       {bench(conv(k30)):7.1f} ms")
print(f"tap loop k3      {bench(taps(k3)):7.1f} ms")
print(f"tap loop k30     {bench(taps(k30)):7.1f} ms")
print(f"ssm loop N=8     {bench(lambda: ssm(x)):7.1f} ms")
