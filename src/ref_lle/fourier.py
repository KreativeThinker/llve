import torch

def amplitude_phase_split(x):
    fft = torch.fft.fft2(x)
    amp = torch.abs(fft)
    pha = torch.angle(fft)
    return amp, pha

def recombine(amp, pha):
    fft = amp * torch.exp(1j * pha)
    x = torch.real(torch.fft.ifft2(fft))
    return x

def zero_frequency_component(amp):
    fft_shifted = torch.fft.fftshift(amp, dim=(-2, -1))
    h, w = amp.shape[-2:]
    center_h, center_w = h // 2, w // 2
    return fft_shifted[..., center_h, center_w]
