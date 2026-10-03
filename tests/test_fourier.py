import torch
import numpy as np
from ref_lle.fourier import amplitude_phase_split, recombine, zero_frequency_component


def test_amplitude_phase_split():
    img = torch.randn(64, 64)
    amp, pha = amplitude_phase_split(img)
    assert amp.shape == img.shape
    assert pha.shape == img.shape
    assert amp.dtype == torch.float32
    assert pha.dtype == torch.float32


def test_round_trip():
    img = torch.rand(64, 64)
    amp, pha = amplitude_phase_split(img)
    reconstructed = recombine(amp, pha)

    assert reconstructed.shape == img.shape
    diff = torch.abs(reconstructed - img).mean()
    assert diff < 0.1, f"Round trip error too large: {diff}"


def test_amplitude_gain():
    img = torch.ones(64, 64) * 0.5
    amp, pha = amplitude_phase_split(img)

    amp_scaled = amp * 2.0
    reconstructed = recombine(amp_scaled, pha)

    brightness_original = img.mean()
    brightness_scaled = reconstructed.mean()
    assert brightness_scaled > brightness_original, "Amplitude scaling should increase brightness"


def test_zfc():
    img = torch.ones(64, 64) * 0.5
    amp, _ = amplitude_phase_split(img)
    zfc = zero_frequency_component(amp)

    assert zfc.shape == torch.Size([1])
    assert zfc > 0


if __name__ == "__main__":
    test_amplitude_phase_split()
    print("✓ test_amplitude_phase_split")

    test_round_trip()
    print("✓ test_round_trip")

    test_amplitude_gain()
    print("✓ test_amplitude_gain")

    test_zfc()
    print("✓ test_zfc")

    print("\nAll fourier tests passed!")
