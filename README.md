# ReF-LLE: Personalized Low-Light Enhancement via Reference-Guided Deep RL

Implementation of "ReF-LLE: Personalized Low-Light Enhancement via Reference-Guided Deep Reinforcement Learning" paper.

## Setup

```bash
uv sync
```

## Project Structure

- `src/ref_lle/` - Main package
  - `fourier.py` - Fourier amplitude/phase decomposition
  - `env.py` - Low-light enhancement environment
  - `rewards.py` - Reward functions (quality + amplitude exposure)
  - `model.py` - CNN policy/value network
  - `a3c.py` - A3C training algorithm
  - `data.py` - Dataset loading (LOL + synthetic fallback)
  - `config.py` - Configuration presets (tiny, paper)
  - `train.py` - Training script
  - `infer.py` - Inference with ZFC-guided adaptation
  - `metrics.py` - Evaluation metrics (PSNR, SSIM, NIQE)
  
- `tests/` - Unit tests

## Quick Start

### Train on synthetic data (CPU)

```bash
uv run python -m ref_lle.train --preset tiny
```

### Inference

```bash
uv run python -m ref_lle.infer --input low_light.png --output enhanced.png --max-iterations 10
```

With reference image:

```bash
uv run python -m ref_lle.infer --input low_light.png --ref normal_light.png --output enhanced.png
```

## Paper Algorithms

### 1. Fourier Amplitude/Phase Split (Eq. 1-2)

Images decomposed into amplitude (brightness) and phase (structure) in Fourier domain.

```
Amp, Pha = |FFT(x)|, angle(FFT(x))
Amp_t = A_t * Amp_{t-1}  (pixel-wise gain)
x_t = iFFT(Amp_t * e^{j Pha_0})
```

where A_t = exp(alpha), alpha ∈ [-0.1, 0.2] (31 discrete actions).

### 2. Zero-Frequency Component (ZFC) Guide

ZFC = center value of fftshift(Amplitude). Target: 2.5e5 (tunable).

```
r_amp = |zfc_target / zfc_current - 1|
```

### 3. Reward Functions (Eq. 3-5)

**Image Quality Reward:**
```
r_iq = S(s_t) - S(s_0)
```
where S(·) = UNIQUE quality score. Currently uses a proxy scorer (contrast + brightness).

**Amplitude Exposure Reward:**
```
r_amp = |zfc_target / zfc_current - 1|
```

**Combined:**
```
r_t = w_iq * r_iq - w_amp * r_amp
```
Hyperparameters: w_iq=1000, w_amp=60

### 4. A3C Architecture

- Shared encoder (3-layer CNN)
- Policy head: per-pixel logits over 31 actions
- Value head: scalar critic
- Async workers, discounted returns (γ=0.95)
- Adam optimizer (lr=0.002), entropy coef=0.01

### 5. ZFC-Guided Inference

Adaptive iteration: policy steps on low-light image until:
1. ZFC target reached (within ±10%)
2. Max iterations reached

User options:
- Reference image (auto-compute ZFC_ref)
- Manual ZFC target
- Iteration count

## Configuration

### Tiny Config (CPU testing)
- 10 rounds, 5 steps/episode, batch 1
- Hidden dim 16, max steps 5

### Paper Config (full training)
- 10,000 rounds, 10 steps/episode, batch 2
- Hidden dim 32, max steps 10

Edit in `config.py`.

## Datasets

### LOL Dataset

Download from [LOL paper repo](https://github.com/weichen582/RetinexNet).
Place in `data/lol/low/` and `data/lol/normal/`.

Configure in training script:

```python
config.lol_train_path = Path("data/lol/low/")
config.lol_test_path = Path("data/lol/normal/")
```

### Synthetic Fallback

If LOL not available, training uses synthetic darkened images (factor=0.3).

## Testing

```bash
uv run pytest tests/ -v
```

Checks:
- Fourier round-trip (< 0.1 MSE)
- ZFC monotonic in amplitude gain
- Environment step shapes
- Reward math correctness

## Deviations from Paper

1. **Architecture**: Paper supplementary missing. We use small CNN (32 channels, 3 conv layers).
2. **Action space**: Discrete 31 actions (alpha step 0.01) instead of continuous.
3. **Reward scorer**: Proxy scorer (contrast + brightness) until UNIQUE weights available.
4. **Per-pixel actions**: Each pixel chooses an action independently (no spatial consistency).
5. **Inference**: ZFC guidance at each step; paper may use final ZFC check.

## Assumptions Documented

- Compute: CPU only. Full training requires GPU.
- Data: Synthetic pairs if LOL unavailable.
- Scorer: Pluggable interface; swap real UNIQUE when weights available.
- Network: Reasonable defaults from similar RL papers.

## Next Steps

1. Swap proxy scorer → real UNIQUE (`pyiqa` or author weights)
2. Add LOL dataset support + full training on GPU
3. Spatial consistency (action smoothing, guided by phase)
4. User study for personalization preferences
5. Comparison with state-of-art baselines

## References

[1] Ming Zhao et al., "ReF-LLE: Personalized Low-Light Enhancement via Reference-Guided Deep Reinforcement Learning", CVPR 2025.
