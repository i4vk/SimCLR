import torch
import random
from torchvision.transforms import functional as F

class AddGaussianNoise(object):
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std

class RandomScale(object):
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*self.scale_range)
        return x * factor

class RandomShift(object):
    def __init__(self, shift_range=(-0.05, 0.05)):
        self.shift_range = shift_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = random.uniform(*self.shift_range)
        return x + shift

class RandomSmoothing(object):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.kernel = torch.ones(1,1,kernel_size) / kernel_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, L)
        x_unsq = x.unsqueeze(0)  # (1,1,L)
        kernel = self.kernel.to(x.device)
        sm = torch.nn.functional.conv1d(x_unsq, kernel, padding=self.pad)
        return sm.squeeze(0)

class SpectralAugment(object):
    def __init__(self, p_noise=0.5, p_scale=0.5, p_shift=0.5, p_smooth=0.3):
        self.noise = AddGaussianNoise()
        self.scale = RandomScale()
        self.shift = RandomShift()
        self.smooth = RandomSmoothing()
        self.probs = {'noise': p_noise, 'scale': p_scale, 'shift': p_shift, 'smooth': p_smooth}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probs['noise']:
            x = self.noise(x)
        if random.random() < self.probs['scale']:
            x = self.scale(x)
        if random.random() < self.probs['shift']:
            x = self.shift(x)
        if random.random() < self.probs['smooth']:
            x = self.smooth(x)
        return x