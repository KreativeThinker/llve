import numpy as np
import torch


def amplitude_phase_split(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)

    b, h, w = x.shape
    amp = np.zeros((b, h, w), dtype=np.float32)
    pha = np.zeros((b, h, w), dtype=np.float32)

    for i in range(b):
        fft = np.fft.fft2(x[i])
        amp[i] = np.abs(fft)
        pha[i] = np.angle(fft)

    return torch.from_numpy(amp), torch.from_numpy(pha)


def recombine(amp, pha):
    if isinstance(amp, torch.Tensor):
        amp = amp.cpu().numpy()
    if isinstance(pha, torch.Tensor):
        pha = pha.cpu().numpy()

    if amp.ndim == 2:
        amp = np.expand_dims(amp, axis=0)
    if pha.ndim == 2:
        pha = np.expand_dims(pha, axis=0)

    b, h, w = amp.shape
    x = np.zeros((b, h, w), dtype=np.float32)

    for i in range(b):
        fft = amp[i] * np.exp(1j * pha[i])
        x[i] = np.real(np.fft.ifft2(fft))

    return torch.from_numpy(x)


def zero_frequency_component(amp):
    if isinstance(amp, torch.Tensor):
        amp_np = amp.cpu().numpy()
    else:
        amp_np = amp

    if amp_np.ndim == 2:
        amp_np = np.expand_dims(amp_np, axis=0)

    b, h, w = amp_np.shape
    zfc = np.zeros(b, dtype=np.float32)

    for i in range(b):
        fft_shifted = np.fft.fftshift(amp_np[i])
        center_h, center_w = h // 2, w // 2
        zfc[i] = fft_shifted[center_h, center_w]

    return torch.from_numpy(zfc)
