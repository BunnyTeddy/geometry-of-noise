# The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning

An interactive marimo notebook exploring the geometric foundations of autonomous (noise-agnostic) generative models.

**Paper**: [arXiv:2602.18428](https://arxiv.org/abs/2602.18428) | [alphaXiv](https://www.alphaxiv.org/abs/2602.18428)
**Authors**: Mojtaba Sahraee-Ardakan, Mauricio Delbracio, Peyman Milanfar (Google)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/BunnyTeddy/geometry-of-noise/blob/main/geometry_of_noise.py)

---

## What This Notebook Covers

Standard diffusion models **need** to know the noise level to work. This paper shows they don't have to. Through 7 interactive sections, you'll discover:

1. **The Paradox** — The marginal energy landscape has an infinitely deep well near data
2. **The Resolution** — Autonomous models implicitly learn a Riemannian metric that tames the singularity
3. **The Stability Rule** — Velocity-prediction works, noise-prediction fails (Jensen Gap)
4. **Novel Extension** — The same principles hold beyond Gaussian noise

## Novel Contribution

This notebook extends the paper's analysis to **non-Gaussian noise distributions** (Laplacian and Gaussian-Laplacian mixture), demonstrating that the Riemannian flow insight is a **general principle**, not a Gaussian artifact.

## Running Locally

```bash
# Run interactively in browser
uvx marimo run geometry_of_noise.py --sandbox

# Edit mode
uvx marimo edit geometry_of_noise.py --sandbox
```

## Requirements

- Python >= 3.11
- marimo, numpy, scipy, matplotlib

All dependencies are auto-installed via PEP 723 inline metadata when using `--sandbox`. No GPU or heavy ML frameworks required — the neural network is implemented in pure numpy for maximum compatibility (including WASM/molab).

## Competition

Built for the **alphaXiv x marimo** "Bring Research to Life" competition (April 2026).
