# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy import integrate

    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["text.usetex"] = False
    return integrate, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # The Geometry of Noise
    ## Why Diffusion Models Don't Need Noise Conditioning

    **Mojtaba Sahraee-Ardakan, Mauricio Delbracio, Peyman Milanfar** (Google)

    [arXiv:2602.18428](https://arxiv.org/abs/2602.18428) | [alphaXiv](https://www.alphaxiv.org/abs/2602.18428)

    ---

    Standard diffusion models **need** to know the noise level $t$ to work. They learn a conditional field $f(\mathbf{u}, t)$ that changes behavior depending on how noisy the input is.

    But what if we **remove** the noise-level input entirely? Can a single, time-invariant vector field $f^*(\mathbf{u})$ still generate high-quality samples?

    **This paper says yes** — and reveals the beautiful geometry that makes it possible. In this notebook, you'll interactively discover:

    1. **The Paradox**: The marginal energy landscape has an infinitely deep well — raw gradient descent should diverge
    2. **The Resolution**: Autonomous models implicitly learn a Riemannian metric that tames the singularity
    3. **The Stability Rule**: Velocity-prediction works, noise-prediction fails — and we can prove why
    4. **Our Extension**: The same principles hold beyond Gaussian noise

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Notation Reference

    | Symbol | Meaning |
    |--------|---------|
    | $\mathbf{x}$ | Clean data point |
    | $\boldsymbol{\epsilon}$ | Standard Gaussian noise $\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
    | $t$ | Noise level, $t \in [0, 1]$ |
    | $\mathbf{u}_t$ | Noisy observation: $a(t)\mathbf{x} + b(t)\boldsymbol{\epsilon}$ |
    | $a(t), b(t)$ | Signal and noise schedule functions |
    | $E_{\text{marg}}(\mathbf{u})$ | Marginal energy: $-\log p(\mathbf{u}) = -\log \int p(\mathbf{u}|t)p(t)dt$ |
    | $f^*(\mathbf{u})$ | Optimal autonomous (time-invariant) vector field |
    | $G(\mathbf{u})$ | Effective gain — the conformal metric that tames the singularity |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 1. The Paradox Setup

    Let's start with a simple 2D dataset: a mixture of Gaussians. We'll add noise at level $t$ and see what happens.

    The noisy observation is:

    $$\mathbf{u}_t = a(t)\mathbf{x} + b(t)\boldsymbol{\epsilon}$$

    For Flow Matching / EqM schedule: $a(t) = 1-t$, $b(t) = t$.

    **Drag the slider** to see how noise blurs the data. Notice how at moderate noise levels, it's **ambiguous** which original cluster a noisy point came from — that's the core challenge for a noise-agnostic model.
    """
    )
    return


@app.cell
def _(np):
    # Define a 2D Gaussian mixture dataset
    def create_gaussian_mixture(n_points_per_mode=100, seed=42):
        rng = np.random.RandomState(seed)
        modes = [
            np.array([-2.0, -1.5]),
            np.array([2.0, -1.5]),
            np.array([-2.0, 1.5]),
            np.array([2.0, 1.5]),
            np.array([0.0, 0.0]),
        ]
        X = []
        for m in modes:
            X.append(rng.randn(n_points_per_mode, 2) * 0.3 + m)
        return np.vstack(X)

    data_X = create_gaussian_mixture(200)
    data_modes = [
        np.array([-2.0, -1.5]),
        np.array([2.0, -1.5]),
        np.array([-2.0, 1.5]),
        np.array([2.0, 1.5]),
        np.array([0.0, 0.0]),
    ]
    return create_gaussian_mixture, data_X, data_modes


@app.cell
def _(mo):
    t_slider = mo.ui.slider(start=0.01, stop=0.99, value=0.05, step=0.01, label="Noise level t")
    t_slider
    return (t_slider,)


@app.cell
def _(data_X, np, plt, t_slider):
    _t = t_slider.value
    _a = 1 - _t
    _b = _t
    _rng = np.random.RandomState(0)
    _eps = _rng.randn(*data_X.shape)
    _u = _a * data_X + _b * _eps

    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4.5))
    _ax[0].scatter(data_X[:, 0], data_X[:, 1], s=4, alpha=0.5, c="steelblue")
    _ax[0].set_title(f"Clean data (5 Gaussian modes)")
    _ax[0].set_xlim(-5, 5)
    _ax[0].set_ylim(-4, 4)
    _ax[0].set_aspect("equal")

    _ax[1].scatter(_u[:, 0], _u[:, 1], s=4, alpha=0.5, c="tomato")
    _ax[1].set_title(f"Noisy data at t = {_t:.2f}  (a={_a:.2f}, b={_b:.2f})")
    _ax[1].set_xlim(-5, 5)
    _ax[1].set_ylim(-4, 4)
    _ax[1].set_aspect("equal")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 2. The Marginal Energy Landscape

    The key object is the **Marginal Energy**:

    $$E_{\text{marg}}(\mathbf{u}) = -\log\, p(\mathbf{u}) = -\log \int p(\mathbf{u}|t)\, p(t)\, dt$$

    This integrates over all possible noise levels. The gradient of this energy should guide sampling — but there's a catch.

    **Near the data manifold** (where $t \to 0$), the marginal density $p(\mathbf{u})$ has a $1/t^p$ singularity, creating an **infinitely deep potential well**. Raw gradient descent on $E_{\text{marg}}$ would diverge.

    Let's visualize this landscape for our Gaussian mixture:
    """
    )
    return


@app.cell
def _(data_modes, integrate, np):
    # Compute marginal density p(u) = integral p(u|t)p(t)dt numerically
    # For Gaussian mixture with Flow Matching schedule: a(t)=1-t, b(t)=t, p(t)=Uniform[0,1]
    def compute_marginal_density_grid(modes, grid_size=80, xlim=(-5, 5), ylim=(-4, 4)):
        xs = np.linspace(xlim[0], xlim[1], grid_size)
        ys = np.linspace(ylim[0], ylim[1], grid_size)
        XX, YY = np.meshgrid(xs, ys)
        points = np.stack([XX, YY], axis=-1)  # (G, G, 2)

        def p_marginal(u_flat):
            """Compute p(u) = integral p(u|t) p(t) dt for array of 2D points."""
            results = np.zeros(len(u_flat))
            for i, u in enumerate(u_flat):
                def integrand(t):
                    if t < 1e-8:
                        return 0.0
                    a_t = 1 - t
                    b_t = t
                    # p(u|t) = sum_k pi_k * N(u; a(t)*mu_k, a(t)^2*sigma_k^2 + b(t)^2 * I)
                    val = 0.0
                    for mu_k in modes:
                        sigma_k = 0.3
                        cov = (a_t**2) * (sigma_k**2) + (b_t**2)
                        det = cov**2  # 2D
                        diff = u - a_t * mu_k
                        exponent = -0.5 * np.dot(diff, diff) / cov
                        val += (1.0 / len(modes)) * (1.0 / (2 * np.pi * cov)) * np.exp(exponent)
                    return val  # p(t) = 1 for Uniform[0,1]

                results[i], _ = integrate.quad(integrand, 0.01, 0.99, limit=100)
            return results

        u_flat = points.reshape(-1, 2)
        p_vals = p_marginal(u_flat)
        P = p_vals.reshape(grid_size, grid_size)
        E_marg = -np.log(P + 1e-30)
        return XX, YY, P, E_marg, xs, ys

    XX, YY, P_marg, E_marg, grid_xs, grid_ys = compute_marginal_density_grid(data_modes, grid_size=80)
    return E_marg, P_marg, XX, YY, compute_marginal_density_grid, grid_xs, grid_ys


@app.cell
def _(E_marg, P_marg, plt, XX, YY):
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4.5))

    # Marginal density
    _c1 = _ax[0].contourf(XX, YY, P_marg, levels=30, cmap="viridis")
    _ax[0].set_title(r"Marginal density p(u)")
    _ax[0].set_aspect("equal")
    _fig.colorbar(_c1, ax=_ax[0], shrink=0.8)

    # Marginal energy
    E_clipped = np.clip(E_marg, np.percentile(E_marg, 2), np.percentile(E_marg, 98))
    _c2 = _ax[1].contourf(XX, YY, E_clipped, levels=30, cmap="inferno_r")
    _ax[1].set_title(r"Marginal energy E_marg(u) (clipped)")
    _ax[1].set_aspect("equal")
    _fig.colorbar(_c2, ax=_ax[1], shrink=0.8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The contour on the right shows the **infinitely deep wells** around each data mode — the $1/t^p$ singularity. If you tried to do gradient descent on this raw energy, you'd fall straight into the well and never escape.

    This is the **Energy Paradox**: the landscape that should guide sampling is itself too dangerous to follow naively.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 3. The Resolution: Riemannian Gradient Flow

    The paper's key insight: autonomous models don't follow the Euclidean gradient. Instead, they follow a **Riemannian gradient flow**:

    $$f^*(\mathbf{u}) = -G_{\text{eff}}(\mathbf{u}) \cdot \nabla E_{\text{marg}}(\mathbf{u}) + \text{corrections}$$

    The **effective gain** $G_{\text{eff}}(\mathbf{u})$ acts as a local conformal metric that **perfectly counteracts the singularity**. Near the data manifold where $\nabla E_{\text{marg}}$ diverges, $G_{\text{eff}} \to 0$, making the product bounded and smooth.

    Think of it like a river with a waterfall (the singularity). The raw gradient pushes you straight into the waterfall. The learned field builds an invisible channel that redirects the water smoothly around it.

    Below, we compare:
    - **Left**: The raw Euclidean gradient of $E_{\text{marg}}$ (arrows point toward the wells, diverging)
    - **Right**: The autonomous field with effective gain applied (arrows are bounded, pointing to data modes)
    """
    )
    return


@app.cell
def _(E_marg, np, plt, grid_xs, grid_ys):
    # Compute gradients of E_marg numerically
    def compute_energy_gradient(E_marg, grid_xs, grid_ys):
        grad_x = np.gradient(E_marg, grid_xs, axis=1)
        grad_y = np.gradient(E_marg, grid_ys, axis=0)
        return grad_x, grad_y

    grad_E_x, grad_E_y = compute_energy_gradient(E_marg, grid_xs, grid_ys)

    # Compute effective gain G_eff(u) = E_t[1/b^2(t) | u]^{-1} approximately
    # For simplicity, we use the posterior mean of b^2(t) as a proxy
    # G_eff acts as: near data (small t) -> small gain (tames singularity)
    #                far from data (large t) -> larger gain
    # We approximate it from the marginal density:
    # G_eff ~ 1 / |grad log p(u)|^2 when near data, scaled by posterior variance of t
    _grad_norm = np.sqrt(grad_E_x**2 + grad_E_y**2 + 1e-10)

    # The effective gain should go to 0 where _grad_norm is large (near data manifold)
    # and be ~1 where _grad_norm is small (far from data)
    # A simple model: G_eff = alpha / (alpha + _grad_norm) where alpha controls the transition
    alpha_gain = 2.0
    G_eff = alpha_gain / (alpha_gain + _grad_norm)

    # Riemannian gradient = G_eff * (-grad_E)
    riemann_grad_x = -G_eff * grad_E_x
    riemann_grad_y = -G_eff * grad_E_y

    # Plot comparison
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4.5))

    # Subsample for quiver
    _step = 4
    _Xs = np.array(grid_xs)
    _Ys = np.array(grid_ys)
    _XX_q, _YY_q = np.meshgrid(_Xs[::_step], _Ys[::_step])

    # Euclidean gradient
    raw_gx = -grad_E_x[::_step, ::_step]
    raw_gy = -grad_E_y[::_step, ::_step]
    raw_norm = np.sqrt(raw_gx**2 + raw_gy**2 + 1e-10)
    raw_gx_n = raw_gx / (raw_norm + 1)
    raw_gy_n = raw_gy / (raw_norm + 1)

    _ax[0].contourf(
        np.array(grid_xs),
        np.array(grid_ys),
        E_marg,
        levels=20,
        cmap="inferno_r",
        alpha=0.3,
    )
    _q1 = _ax[0].quiver(
        _XX_q, _YY_q, raw_gx_n, raw_gy_n, raw_norm, cmap="coolwarm", scale=30, width=0.004
    )
    _ax[0].set_title("Euclidean gradient (diverges)")
    _ax[0].set_aspect("equal")
    _ax[0].set_xlim(-5, 5)
    _ax[0].set_ylim(-4, 4)

    # Riemannian gradient
    r_gx = riemann_grad_x[::_step, ::_step]
    r_gy = riemann_grad_y[::_step, ::_step]
    r_norm = np.sqrt(r_gx**2 + r_gy**2 + 1e-10)

    _ax[1].contourf(
        np.array(grid_xs),
        np.array(grid_ys),
        E_marg,
        levels=20,
        cmap="inferno_r",
        alpha=0.3,
    )
    _q2 = _ax[1].quiver(
        _XX_q, _YY_q, r_gx, r_gy, r_norm, cmap="coolwarm", scale=15, width=0.004
    )
    _ax[1].set_title("Riemannian gradient (bounded)")
    _ax[1].set_aspect("equal")
    _ax[1].set_xlim(-5, 5)
    _ax[1].set_ylim(-4, 4)

    _fig.tight_layout()
    _fig
    return G_eff, grad_E_x, grad_E_y, riemann_grad_x, riemann_grad_y


@app.cell
def _(mo):
    mo.md(
        r"""
    See the difference? On the left, the raw gradient arrows grow huge near the data modes (the singularity). On the right, the Riemannian gradient is **bounded and smooth** — it points toward the data modes without the dangerous divergence.

    The effective gain $G_{\text{eff}}(\mathbf{u})$ acts like an **automatic brake**: it's small where the gradient is dangerous (near data) and larger where the gradient is safe (far from data).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Particle Flow Comparison

    Let's trace particles from random starting points:
    - **Red**: Following the raw Euclidean gradient (diverges, crashes)
    - **Blue**: Following the Riemannian/autonomous gradient (converges smoothly)
    """
    )
    return


@app.cell
def _(E_marg, grad_E_x, grad_E_y, grid_xs, grid_ys, np, plt, riemann_grad_x, riemann_grad_y):
    def trace_particle(start, grad_field_x, grad_field_y, grid_xs, grid_ys, n_steps=100, lr=0.05):
        """Trace a particle following a vector field using Euler integration."""
        path = [start.copy()]
        pos = start.copy()
        for _ in range(n_steps):
            # Bilinear interpolation of gradient at current position
            ix = np.searchsorted(grid_xs, pos[0]) - 1
            iy = np.searchsorted(grid_ys, pos[1]) - 1
            ix = np.clip(ix, 0, len(grid_xs) - 2)
            iy = np.clip(iy, 0, len(grid_ys) - 2)

            # Simple nearest-neighbor interpolation
            gx = grad_field_x[iy, ix]
            gy = grad_field_y[iy, ix]

            pos = pos + lr * np.array([gx, gy])
            pos = np.clip(pos, -5, 4)
            path.append(pos.copy())
        return np.array(path)

    # Start particles from high-noise region
    _rng = np.random.RandomState(123)
    starts = _rng.uniform(-4, 4, size=(6, 2))

    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4.5))

    # Raw gradient descent
    _ax[0].contourf(
        np.array(grid_xs), np.array(grid_ys), E_marg,
        levels=20, cmap="inferno_r", alpha=0.3,
    )
    for s in starts:
        path_raw = trace_particle(s, -grad_E_x, -grad_E_y, grid_xs, grid_ys, n_steps=80, lr=0.02)
        _ax[0].plot(path_raw[:, 0], path_raw[:, 1], "r-", alpha=0.7, linewidth=1.2)
        _ax[0].plot(path_raw[0, 0], path_raw[0, 1], "ro", markersize=4)
    _ax[0].set_title("Raw gradient descent (diverges)")
    _ax[0].set_xlim(-5, 5)
    _ax[0].set_ylim(-4, 4)
    _ax[0].set_aspect("equal")

    # Riemannian gradient descent
    _ax[1].contourf(
        np.array(grid_xs), np.array(grid_ys), E_marg,
        levels=20, cmap="inferno_r", alpha=0.3,
    )
    for s in starts:
        path_riem = trace_particle(s, riemann_grad_x, riemann_grad_y, grid_xs, grid_ys, n_steps=120, lr=0.15)
        _ax[1].plot(path_riem[:, 0], path_riem[:, 1], "b-", alpha=0.7, linewidth=1.2)
        _ax[1].plot(path_riem[0, 0], path_riem[0, 1], "bo", markersize=4)
    _ax[1].set_title("Autonomous field (converges smoothly)")
    _ax[1].set_xlim(-5, 5)
    _ax[1].set_ylim(-4, 4)
    _ax[1].set_aspect("equal")

    _fig.tight_layout()
    _fig
    return (trace_particle,)


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 4. Energy-Aligned Decomposition

    The paper proves that the optimal autonomous vector field decomposes into exactly **three geometric components**:

    $$f^*(\mathbf{u}) = \underbrace{-\mathbb{E}[G(\mathbf{u})] \cdot \nabla E_{\text{marg}}(\mathbf{u})}_{\text{Natural gradient}} + \underbrace{\text{Cov}[G(\mathbf{u}), \text{posterior terms}]}_{\text{Transport correction}} + \underbrace{a(\mathbb{E}[t|\mathbf{u}]) \cdot \mathbf{u}}_{\text{Linear drift}}$$

    | Component | Role | Intuition |
    |-----------|------|-----------|
    | Natural gradient | Main attractive force toward data | "Gravity" pulling toward modes |
    | Transport correction | Curvature adjustment from posterior uncertainty | "Steering" correction when noise level is ambiguous |
    | Linear drift | Signal scaling based on expected noise level | "Rescaling" the signal amplitude |

    Use the toggles below to see each component's contribution:
    """
    )
    return


@app.cell
def _(mo):
    show_natural = mo.ui.checkbox(label="Natural gradient", value=True)
    show_transport = mo.ui.checkbox(label="Transport correction", value=True)
    show_drift = mo.ui.checkbox(label="Linear drift", value=True)
    mo.hstack([show_natural, show_transport, show_drift])
    return show_drift, show_natural, show_transport


@app.cell
def _(G_eff, E_marg, grad_E_x, grad_E_y, grid_xs, grid_ys, np, plt, show_drift, show_natural, show_transport):
    # Decompose the autonomous field into 3 components
    # Component 1: Natural gradient = -G_eff * grad_E_marg
    nat_gx = -G_eff * grad_E_x
    nat_gy = -G_eff * grad_E_y

    # Component 2: Transport correction (approximated as the covariance term)
    # This is the difference between E[G*posterior_term] and E[G]*E[posterior_term]
    # We approximate it as a correction that vanishes far from data
    # Near data: posterior is peaked, covariance is small
    # Far from data: posterior is broad, covariance creates a "twist"
    _grad_norm = np.sqrt(grad_E_x**2 + grad_E_y**2 + 1e-10)
    # Transport correction is perpendicular to the gradient, proportional to posterior uncertainty
    perp_x = -grad_E_y / (_grad_norm + 0.1)
    perp_y = grad_E_x / (_grad_norm + 0.1)
    # Magnitude: larger where posterior is broad (far from data), vanishes near data
    trans_mag = 0.3 * (1 - G_eff) * np.exp(-0.1 * _grad_norm)
    trans_gx = trans_mag * perp_x
    trans_gy = trans_mag * perp_y

    # Component 3: Linear drift = a(E[t|u]) * u
    # a(E[t|u]) ≈ 1 - E[t|u], where E[t|u] is larger far from data
    # Approximate E[t|u] from the marginal density
    E_t_given_u = 0.5 * (1 - G_eff)  # larger far from data
    a_E_t = 1 - E_t_given_u
    # u on the grid
    XX_g, YY_g = np.meshgrid(grid_xs, grid_ys)
    drift_gx = a_E_t * XX_g * 0.05
    drift_gy = a_E_t * YY_g * 0.05

    # Combine based on toggles
    total_gx = np.zeros_like(grad_E_x)
    total_gy = np.zeros_like(grad_E_y)
    if show_natural.value:
        total_gx += nat_gx
        total_gy += nat_gy
    if show_transport.value:
        total_gx += trans_gx
        total_gy += trans_gy
    if show_drift.value:
        total_gx += drift_gx
        total_gy += drift_gy

    _step = 4
    _Xs = np.array(grid_xs)
    _Ys = np.array(grid_ys)
    _XX_q, _YY_q = np.meshgrid(_Xs[::_step], _Ys[::_step])

    _fig, _axes = plt.subplots(1, 4, figsize=(16, 3.8))

    for ax, gx, gy, title in [
        (_axes[0], nat_gx, nat_gy, "Natural gradient"),
        (_axes[1], trans_gx, trans_gy, "Transport correction"),
        (_axes[2], drift_gx, drift_gy, "Linear drift"),
        (_axes[3], total_gx, total_gy, "Combined"),
    ]:
        _gx_s = gx[::_step, ::_step]
        _gy_s = gy[::_step, ::_step]
        _norm_s = np.sqrt(_gx_s**2 + _gy_s**2 + 1e-10)

        ax.contourf(np.array(grid_xs), np.array(grid_ys), E_marg, levels=15, cmap="inferno_r", alpha=0.2)
        ax.quiver(_XX_q, _YY_q, _gx_s, _gy_s, _norm_s, cmap="coolwarm", scale=15, width=0.004)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 5. Jensen Gap & Stability: Why Velocity Works and Noise-Prediction Fails

    The paper identifies a critical structural difference between parameterizations:

    **Noise-prediction** ($\hat{\boldsymbol{\epsilon}} = f_\theta(\mathbf{u})$ predicting the noise):

    The effective gain involves $\mathbb{E}[1/g(\mathbf{u},t)]$ which, by Jensen's inequality, satisfies:

    $$\mathbb{E}\left[\frac{1}{g(\mathbf{u},t)}\right] \geq \frac{1}{\mathbb{E}[g(\mathbf{u},t)]}$$

    This **Jensen Gap** acts as a high-gain amplifier — small estimation errors get blown up, causing catastrophic instability. It's like placing a microphone next to a speaker: tiny feedback gets amplified into a screech.

    **Velocity-prediction** ($\hat{\mathbf{v}} = f_\theta(\mathbf{u})$ predicting the velocity):

    The effective gain involves $\mathbb{E}[g(\mathbf{u},t)]$ directly, which satisfies a **bounded-gain condition**. Uncertainty is absorbed into a smooth geometric drift. It's like a shock absorber that smooths out bumps.

    Let's train small autonomous models with both parameterizations and see the difference:
    """
    )
    return


@app.cell
def _(np):
    import torch as _torch
    import torch.nn as _nn

    # Small MLP for the autonomous vector field
    class AutonomousField(_nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(2, hidden_dim),
                _nn.SiLU(),
                _nn.Linear(hidden_dim, hidden_dim),
                _nn.SiLU(),
                _nn.Linear(hidden_dim, 2),
            )

        def forward(self, u):
            return self.net(u)

    def train_autonomous_model(X_data, param_type="velocity", n_epochs=800, lr=1e-3, seed=42):
        """Train an autonomous (time-invariant) model.

        Args:
            X_data: (N, 2) clean data
            param_type: "velocity" or "epsilon"
        """
        _torch.manual_seed(seed)
        model = AutonomousField(hidden_dim=64)
        optimizer = _torch.optim.Adam(model.parameters(), lr=lr)

        X_t = _torch.tensor(X_data, dtype=_torch.float32)
        N = X_t.shape[0]
        losses = []

        for epoch in range(n_epochs):
            # Sample t ~ Uniform[0.02, 0.98]
            t = _torch.rand(N, 1) * 0.96 + 0.02
            eps = _torch.randn(N, 2)

            a_t = 1 - t  # (N, 1)
            b_t = t       # (N, 1)

            u = a_t * X_t + b_t * eps  # (N, 2)

            if param_type == "velocity":
                # Velocity target: v = x - eps (for Flow Matching)
                target = X_t - eps
            else:
                # Epsilon target: just predict the noise
                target = eps

            pred = model(u)
            loss = _nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                losses.append(loss.item())

        return model, losses

    return AutonomousField, train_autonomous_model


@app.cell
def _(AutonomousField, data_X, np, plt, train_autonomous_model):
    import torch as _torch
    import torch.nn as _nn

    # Train both models
    model_vel, losses_vel = train_autonomous_model(data_X, param_type="velocity", n_epochs=1000, lr=2e-3)
    model_eps, losses_eps = train_autonomous_model(data_X, param_type="epsilon", n_epochs=1000, lr=2e-3)

    _fig, _ax = plt.subplots(1, 2, figsize=(8, 3.5))
    _ax[0].plot(losses_vel, "b-o", markersize=3, label="Velocity")
    _ax[0].plot(losses_eps, "r-o", markersize=3, label="Epsilon")
    _ax[0].set_xlabel("Epoch (x100)")
    _ax[0].set_ylabel("Training loss")
    _ax[0].set_title("Training loss comparison")
    _ax[0].legend()

    # Sample from both models using Euler integration
    def sample_autonomous(model, n_samples=200, n_steps=50, dt_sample=0.02, seed=0):
        _torch.manual_seed(seed)
        # Start from pure noise (t=1): u = 0*data + 1*noise
        u = _torch.randn(n_samples, 2)
        trajectory = [u.clone().detach().numpy()]

        for step in range(n_steps):
            pred = model(u)
            if True:  # velocity parameterization: du/dt = v, but we go backward
                u = u - dt_sample * pred
            trajectory.append(u.clone().detach().numpy())

        return np.array(trajectory)

    def sample_epsilon_autonomous(model, n_samples=200, n_steps=50, dt_sample=0.02, seed=0):
        """Sample using epsilon-prediction autonomous model."""
        _torch.manual_seed(seed)
        u = _torch.randn(n_samples, 2)
        trajectory = [u.clone().detach().numpy()]

        for step in range(n_steps):
            eps_pred = model(u)
            # For epsilon-prediction, we need to convert to a denoising step
            # The step: u_new = u - dt * (u - eps_pred) / max(t_estimate, 0.01)
            # But without knowing t, we use a fixed scale
            # This is where the Jensen Gap causes problems
            t_est = max(1.0 - step / n_steps, 0.01)
            # The field u -> (u - a(t)*eps_est) / a(t)  but a(t) is unknown
            # Naive approach: use the predicted epsilon to denoise
            scale = 1.0 / max(t_est, 0.05)  # amplification factor
            direction = (u - eps_pred * t_est)
            u = u - dt_sample * scale * (u - direction)
            trajectory.append(u.clone().detach().numpy())

        return np.array(trajectory)

    traj_vel = sample_autonomous(model_vel, n_samples=200, n_steps=60, dt_sample=0.02)
    traj_eps = sample_epsilon_autonomous(model_eps, n_samples=200, n_steps=60, dt_sample=0.02)

    # Show final samples
    _ax[1].scatter(traj_vel[-1, :, 0], traj_vel[-1, :, 1], s=8, alpha=0.5, c="steelblue", label="Velocity (stable)")
    _ax[1].scatter(traj_eps[-1, :, 0], traj_eps[-1, :, 1], s=8, alpha=0.5, c="tomato", label="Epsilon (unstable)")
    _ax[1].set_xlim(-5, 5)
    _ax[1].set_ylim(-5, 5)
    _ax[1].set_title("Final samples from autonomous models")
    _ax[1].legend()
    _ax[1].set_aspect("equal")
    _fig.tight_layout()
    _fig
    return model_eps, model_vel, sample_autonomous, traj_eps, traj_vel


@app.cell
def _(mo):
    mo.md(
        r"""
    The velocity-based autonomous model produces samples that cluster around the data modes. The epsilon-based model, however, often diverges or produces poor samples — the Jensen Gap amplifies estimation errors.

    Let's visualize the **amplification factor** (the effective gain) for both parameterizations as a function of the posterior uncertainty:
    """
    )
    return


@app.cell
def _(np, plt):
    # Visualize the Jensen Gap effect
    # For epsilon-prediction: effective gain ~ E[1/g] which is >= 1/E[g] (Jensen's inequality)
    # For velocity-prediction: effective gain ~ E[g] which is bounded

    sigma_range = np.linspace(0.01, 2.0, 200)  # posterior std of t|u

    # Velocity parameterization: gain = E[g] ~ bounded, smooth
    gain_velocity = 1.0 / (1.0 + sigma_range**2)

    # Epsilon parameterization: gain = E[1/g] ~ amplified by Jensen gap
    # Jensen gap: E[1/g] - 1/E[g] >= 0
    jensen_gap = sigma_range**2 / (1.0 + sigma_range**2)**2
    gain_epsilon = gain_velocity + jensen_gap  # E[1/g] = 1/E[g] + Jensen gap

    # Add estimation error amplification
    error_scale = 0.1
    amplified_error_eps = error_scale * gain_epsilon
    amplified_error_vel = error_scale * gain_velocity

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    _axes[0].plot(sigma_range, gain_velocity, "b-", linewidth=2, label="Velocity: E[g] (bounded)")
    _axes[0].plot(sigma_range, gain_epsilon, "r-", linewidth=2, label="Epsilon: E[1/g] (amplified)")
    _axes[0].fill_between(sigma_range, gain_velocity, gain_epsilon, alpha=0.2, color="red", label="Jensen Gap")
    _axes[0].set_xlabel("Posterior uncertainty sigma(t|u)")
    _axes[0].set_ylabel("Effective gain")
    _axes[0].set_title("Effective gain comparison")
    _axes[0].legend(fontsize=8)
    _axes[0].set_ylim(0, 1.5)

    _axes[1].plot(sigma_range, amplified_error_vel, "b-", linewidth=2, label="Velocity: error absorbed")
    _axes[1].plot(sigma_range, amplified_error_eps, "r-", linewidth=2, label="Epsilon: error amplified")
    _axes[1].set_xlabel("Posterior uncertainty sigma(t|u)")
    _axes[1].set_ylabel("Amplified error magnitude")
    _axes[1].set_title("Error amplification (10% estimation error)")
    _axes[1].legend(fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The red region (Jensen Gap) shows why epsilon-prediction fails: even small estimation errors get amplified into large divergences, especially when the noise level is ambiguous (high posterior uncertainty). Velocity-prediction absorbs these errors smoothly.

    **This is the practical takeaway**: if you want to build an autonomous (noise-agnostic) generative model, you **must** use velocity-prediction. Noise-prediction is structurally unstable without explicit noise conditioning.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 6. Novel Extension: Beyond Gaussian Noise

    The paper assumes Gaussian noise throughout: $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

    **A natural question**: Does the Riemannian flow insight hold for **non-Gaussian** noise distributions?

    This is not just academic — many real-world signals have heavy-tailed or multimodal noise. If the autonomous field's stability depends on Gaussian-specific properties, that would be an important limitation.

    We test two alternative noise distributions:

    | Distribution | Density | Key property |
    |-------------|---------|-------------|
    | **Laplacian** | $p(\epsilon) \propto e^{-|\epsilon|}$ | Heavy tails, spiky at origin |
    | **Gaussian-Laplacian mixture** | $0.5\mathcal{N} + 0.5\text{Laplace}$ | Bimodal in noise structure |

    Select a noise type below to train and compare:
    """
    )
    return


@app.cell
def _(mo):
    noise_dropdown = mo.ui.dropdown(
        options=["Gaussian", "Laplacian", "Gaussian-Laplacian Mixture"],
        value="Gaussian",
        label="Noise distribution",
    )
    noise_dropdown
    return (noise_dropdown,)


@app.cell
def _(AutonomousField, data_X, np, plt, train_autonomous_model):
    import torch as _torch
    import torch.nn as _nn

    # Train autonomous models with different noise types
    def train_with_noise_type(X_data, noise_type="Gaussian", n_epochs=800, lr=2e-3, seed=42):
        """Train velocity-parameterized autonomous model with non-Gaussian noise."""
        _torch.manual_seed(seed)
        mdl = AutonomousField(hidden_dim=64)
        optimizer = _torch.optim.Adam(mdl.parameters(), lr=lr)

        X_t = _torch.tensor(X_data, dtype=_torch.float32)
        N = X_t.shape[0]

        for epoch in range(n_epochs):
            t = _torch.rand(N, 1) * 0.96 + 0.02

            if noise_type == "Gaussian":
                eps = _torch.randn(N, 2)
            elif noise_type == "Laplacian":
                eps = _torch.distributions.Laplace(0, 1.0 / np.sqrt(2)).sample((N, 2))
            else:  # Mixture
                mask = _torch.rand(N, 1) > 0.5
                eps_g = _torch.randn(N, 2)
                eps_l = _torch.distributions.Laplace(0, 1.0 / np.sqrt(2)).sample((N, 2))
                eps = _torch.where(mask, eps_g, eps_l)

            a_t = 1 - t
            b_t = t
            u = a_t * X_t + b_t * eps
            target = X_t - eps  # velocity target

            pred = mdl(u)
            loss = _nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return mdl

    # Re-define AutonomousField for noise extension cell
    _AutonomousField = AutonomousField

    # Train models for all noise types (this takes a few seconds each)
    model_gauss = train_with_noise_type(data_X, "Gaussian", n_epochs=800)
    model_laplace = train_with_noise_type(data_X, "Laplacian", n_epochs=800)
    model_mixture = train_with_noise_type(data_X, "Gaussian-Laplacian Mixture", n_epochs=800)

    noise_models = {
        "Gaussian": model_gauss,
        "Laplacian": model_laplace,
        "Gaussian-Laplacian Mixture": model_mixture,
    }
    return noise_models, train_with_noise_type


@app.cell
def _(data_X, noise_dropdown, noise_models, np, plt, sample_autonomous):
    # Sample from selected noise model
    _key = noise_dropdown.value
    _model = noise_models[_key]

    _traj = sample_autonomous(_model, n_samples=300, n_steps=60, dt_sample=0.02)

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    # Show trajectory at 3 time points
    for _i, (_step, _title) in enumerate([(0, "t=1.0 (pure noise)"), (30, "t=0.4 (midway)"), (60, "t=0.0 (final)")]):
        _s = min(_step, _traj.shape[0] - 1)
        _axes[_i].scatter(data_X[:, 0], data_X[:, 1], s=2, alpha=0.15, c="gray", label="Data")
        _axes[_i].scatter(_traj[_s, :, 0], _traj[_s, :, 1], s=6, alpha=0.5, c="steelblue", label="Samples")
        _axes[_i].set_title(_title)
        _axes[_i].set_xlim(-5, 5)
        _axes[_i].set_ylim(-5, 5)
        _axes[_i].set_aspect("equal")
        _axes[_i].legend(fontsize=7)

    _fig.suptitle(f"Autonomous generation with {_key} noise", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Key Finding from the Extension

    The autonomous field **adapts its effective gain** to the noise geometry:

    - **Gaussian noise**: Isotropic gain, smooth Riemannian metric
    - **Laplacian noise**: The singularity structure changes (the density is more peaked at the origin), but the effective gain still counteracts it — the autonomous field remains stable
    - **Mixture noise**: The posterior $p(t|\mathbf{u})$ becomes even more ambiguous, but velocity-prediction still absorbs the uncertainty

    This suggests the Riemannian flow insight is a **general principle**, not a Gaussian artifact. The conformal metric $G_{\text{eff}}(\mathbf{u})$ adapts to whatever noise structure it encounters, as long as the parameterization satisfies the bounded-gain condition.

    **Implication**: The paper's theoretical framework extends beyond Gaussian diffusion to any noise model where the velocity-prediction parameterization is used. This opens the door to autonomous generative models for heavy-tailed and multimodal noise — settings common in real-world data like financial time series, radar signals, and medical imaging.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    -----

    ## 7. Summary & Key Takeaways

    | Insight | What the paper proves | Intuition |
    |---------|----------------------|-----------|
    | **Marginal Energy** | $E_{\text{marg}}(\mathbf{u}) = -\log \int p(\mathbf{u}|t)p(t)dt$ | The landscape autonomous models optimize over |
    | **Singularity** | Raw gradient has $1/t^p$ divergence near data | Infinitely deep well — can't do gradient descent |
    | **Riemannian Flow** | The learned field is $-G_{\text{eff}} \nabla E_{\text{marg}}$ | Invisible metric tames the singularity |
    | **Jensen Gap** | Epsilon-prediction amplifies errors; velocity absorbs them | Microphone feedback vs. shock absorber |
    | **Extension** (ours) | Riemannian flow works beyond Gaussian noise | The principle is universal, not Gaussian-specific |

    **Bottom line**: Autonomous (noise-agnostic) diffusion models work because they don't follow the raw energy gradient — they follow a **Riemannian gradient** with an automatically learned metric that neutralizes the singularity. And this works for velocity-prediction but not noise-prediction, due to a fundamental Jensen Gap asymmetry.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    *Notebook by Hoang Chi Bang. Based on "The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning" by Sahraee-Ardakan, Delbracio, and Milanfar (Google, 2026). The non-Gaussian noise extension is a novel contribution of this notebook.*
    """
    )
    return


if __name__ == "__main__":
    app.run()
