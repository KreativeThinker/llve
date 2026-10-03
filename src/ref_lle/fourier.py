import numpy as np
import torch
from .utils import ensure_3d


def amplitude_phase_split(x):
    x = ensure_3d(x)
    b, h, w = x.shape
    amp = np.zeros((b, h, w), dtype=np.float32)
    pha = np.zeros((b, h, w), dtype=np.float32)

    for i in range(b):
        fft = np.fft.fft2(x[i])
        amp[i] = np.abs(fft)
        pha[i] = np.angle(fft)

    return torch.from_numpy(amp), torch.from_numpy(pha)


def recombine(amp, pha):
    amp = ensure_3d(amp)
    pha = ensure_3d(pha)

    b, h, w = amp.shape
    x = np.zeros((b, h, w), dtype=np.float32)

    for i in range(b):
        fft = amp[i] * np.exp(1j * pha[i])
        x[i] = np.real(np.fft.ifft2(fft))

    return torch.from_numpy(x)


def zero_frequency_component(amp):
    amp = ensure_3d(amp)
    b, h, w = amp.shape
    zfc = np.zeros(b, dtype=np.float32)

    for i in range(b):
        fft_shifted = np.fft.fftshift(amp[i])
        center_h, center_w = h // 2, w // 2
        zfc[i] = fft_shifted[center_h, center_w]

    return torch.from_numpy(zfc)
