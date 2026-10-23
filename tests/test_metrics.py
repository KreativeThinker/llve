import torch
from ref_lle.metrics import psnr, psnr_y, ssim, niqe


def test_psnr_identical():
    img = torch.rand(3, 64, 64)
    val = psnr(img, img)
    assert val == float('inf') or val > 90.0


def test_psnr_different():
    img1 = torch.zeros(3, 64, 64)
    img2 = torch.ones(3, 64, 64)
    val = psnr(img1, img2)
    assert 0.0 <= val < 10.0


def test_psnr_y():
    img1 = torch.rand(3, 64, 64)
    img2 = img1.clone()
    val = psnr_y(img1, img2)
    assert val == float('inf') or val > 90.0


def test_ssim_identical():
    img = torch.rand(3, 64, 64)
    val = ssim(img, img)
    assert abs(val - 1.0) < 1e-3


def test_ssim_different():
    img1 = torch.zeros(3, 64, 64)
    img2 = torch.ones(3, 64, 64)
    val = ssim(img1, img2)
    assert val < 0.1
