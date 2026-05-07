# Diffusion Model — AI Image Generation with MNIST 🔥

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Major Assignment — Notebook + Written Report

---

## Problem Statement

Diffusion models are the technology behind DALL-E, Stable Diffusion, and Midjourney. This assignment required building and training a class-conditioned Denoising Diffusion Probabilistic Model (DDPM) from scratch — learning the mathematics, architecture, and training process that powers modern generative AI.

## Dataset

**MNIST** — 70,000 handwritten digit images (0–9)
- Image size: 28×28 pixels, grayscale
- Training: 48,000 | Validation: 12,000 (80/20 split)
- Normalized to `[-1, 1]` range

> Downloads automatically via `torchvision.datasets.MNIST`

## Approach & Methodology

### How Diffusion Models Work

```
Forward process:  Clean image  →  Add noise gradually  →  Pure noise
Reverse process:  Pure noise  →  Predict & remove noise →  Clean image
```

The model learns to reverse the noise process — given a noisy image at timestep `t`, predict the noise that was added so it can be subtracted.

### Model Architecture — U-Net

```
Input (1×28×28)
→ InitConv: GELUConvBlock(1, 32)
→ DownBlock(32→64) + RearrangePool (28×28 → 14×14)
→ DownBlock(64→128) + RearrangePool (14×14 → 7×7)
→ MidBlock1(128) + Time Embedding + Class Embedding
→ MidBlock2(128)
→ UpBlock(128→64) + skip (7×7 → 14×14)
→ UpBlock(64→32) + skip (14×14 → 28×28)
→ FinalConv(32→1)
```

**Total parameters:** 1,510,529 (5.8 MB)

### Conditioning

- **Time embedding:** Sinusoidal positional embeddings → Linear → injected at bottleneck
- **Class conditioning:** One-hot digit labels → EmbedBlock → injected at bottleneck alongside time embedding
- This allows the model to generate a specific digit on request

### Noise Schedule

```python
n_steps = 100
beta = torch.linspace(0.0001, 0.02, n_steps)   # Linear schedule
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)          # Cumulative product
```

### Training

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | MSE on predicted noise |
| Epochs | 30 |
| Batch size | 64 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | patience=10 |
| Gradient clipping | max_norm=1.0 |
| GPU | Tesla T4 (15.6 GB) |

## Results & Evaluation

| Metric | Value |
|--------|-------|
| Starting train loss | 0.1068 |
| Final train loss | 0.0565 |
| Best validation loss | **0.0555** |
| Loss improvement | **47.2%** |
| Training epochs | 30 |

### CLIP Evaluation
- Integrated **OpenAI CLIP (ViT-B/32)** to score generated images against text prompts
- 3-prompt evaluation: "handwritten digit X", "clear digit X", "blurry number"
- Simpler digits (1, 7) scored higher on clarity; complex shapes (0, 6, 8) remained harder to generate cleanly

### Visual Progression
Images became recognizable around steps 40–60 of the reverse diffusion process. By step 0, most digits showed clear stroke structure, though some remained stylistically varied.

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
einops>=0.6.0
numpy
matplotlib
Pillow
```

For CLIP evaluation:
```
clip @ git+https://github.com/openai/CLIP.git
ftfy
regex
tqdm
```

Install with:
```bash
pip install torch torchvision einops numpy matplotlib Pillow
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

> **Note:** GPU strongly recommended. Training took ~30–45 minutes on a Tesla T4.

## Files

| File | Description |
|------|-------------|
| `MD_Notebook_Munachimso_Benson_ITAI2376.ipynb` | Full training notebook with architecture, noise visualization, generation, and CLIP evaluation |
| `MD_Report_Munachimso_Benson_ITAI2376.docx` | Written analysis report covering methodology, results, and reflection |

## Learning Outcomes

- Understood the mathematics of the forward and reverse diffusion processes
- Built a complete U-Net with skip connections, time embeddings, and class conditioning
- Learned how noise schedules (linear vs non-linear) affect training dynamics
- Used CLIP as an automated evaluation metric for generative model quality
- Recognized the relationship between diffusion models and products like DALL-E and Stable Diffusion

---

*ITAI-2376 · Spring 2026 · Houston Community College*
