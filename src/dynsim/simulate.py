"""Discretisation, simulation and analytical FRFs for linear state-space models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm


def van_loan_discretisation(
    F: ArrayLike,
    q: float | ArrayLike,
    L: ArrayLike,
    dt: float,
    B: ArrayLike | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """
    Van Loan discretisation of a linear SDE with optional deterministic input.

        dx = F x dt + B u dt + L dβ,    E[dβ dβᵀ] = q dt

    i.e. dβ has spectral density q. Inputs are assumed constant over each sample (zero-order hold).

    Args:
        F: Continuous-time drift (state) matrix, shape (D, D).
        q: Spectral density of the white noise(s): a scalar (times identity) or a matrix.
        L: Matrix (or vector) showing where the noise enters the states, shape (D, W).
        dt: Sample time.
        B: Optional continuous-time input matrix, shape (D, U).

    Returns:
        A: Discrete state-transition matrix, shape (D, D).
        Q: Discrete process-noise covariance, shape (D, D).
        Bd: Discrete input matrix, shape (D, U), or None if `B` was not given.

    Note:
        This always returns three values. Earlier versions returned only (A, Q) when `B` was None.
    """
    F = np.atleast_2d(F)
    D = F.shape[0]

    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L[:, None]

    q = q * np.eye(L.shape[1]) if np.isscalar(q) else np.atleast_2d(q)

    phi = np.block([[F, L @ q @ L.T], [np.zeros_like(F), -F.T]])
    sigma = expm(phi * dt)

    A = sigma[:D, :D]
    Q = sigma[:D, D:] @ A.T

    Bd = None
    if B is not None:
        B = np.atleast_2d(B)
        psi = np.block([[F, B], [np.zeros((B.shape[1], D + B.shape[1]))]])
        Bd = expm(psi * dt)[:D, D:]

    return A, Q, Bd


def simulate_discrete_ssm(
    A: NDArray,
    C: NDArray,
    Q: NDArray,
    R: NDArray,
    x0: NDArray,
    B: NDArray | None = None,
    u: NDArray | None = None,
    T: int = 1000,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Simulate a discrete-time state-space model with optional inputs.

        x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0, Q)
        y_t     = C x_t + v_t,            v_t ~ N(0, R)

    Args:
        A: State-transition matrix, shape (Dx, Dx).
        C: Observation matrix, shape (Dy, Dx).
        Q: Process-noise covariance, shape (Dx, Dx).
        R: Observation-noise covariance, shape (Dy, Dy). Must be positive definite.
        x0: Initial state x_0, shape (Dx,).
        B: Input matrix, shape (Dx, U). Used with `u`.
        u: Inputs, shape (U, T).
        T: Number of time steps. Defaults to 1000.
        rng: Seed or numpy Generator for reproducible noise. Defaults to fresh entropy.

    Returns:
        x: States x_0 ... x_{T-1}, shape (Dx, T).
        y: Observations y_0 ... y_{T-1}, shape (Dy, T), so ``y[:, t]`` is the noisy observation
            of ``x[:, t]``.

    Note:
        Earlier versions returned the states x_1 ... x_T, one step ahead of the observations.
        States and observations now share a time index.
    """
    rng = np.random.default_rng(rng)
    Dx, Dy = A.shape[0], C.shape[0]

    LQ = np.linalg.cholesky(Q + 1e-12 * np.eye(Dx))  # jitter: Q from Van Loan can be near-singular
    LR = np.linalg.cholesky(R)

    w = LQ @ rng.standard_normal((Dx, T))
    v = LR @ rng.standard_normal((Dy, T))

    Bu = np.zeros((Dx, T))
    if B is not None and u is not None:
        Bu = B @ u

    x = np.empty((Dx, T))
    x[:, 0] = x0
    for t in range(T - 1):
        x[:, t + 1] = A @ x[:, t] + Bu[:, t] + w[:, t]

    return x, C @ x + v


def build_frf(
    ws: NDArray, wn: NDArray, zn: NDArray, Phi: NDArray, frf_type: str = "accelerance"
) -> NDArray:
    """
    Analytical frequency response functions from modal properties.

    Warning:
        Frequencies must be in **rad/s**. `SystemMatrices.modal` returns natural frequencies in
        Hz, so pass ``2 * np.pi * modal.wn`` for `wn`.

    Args:
        ws: Frequency lines to evaluate, shape (N,), rad/s.
        wn: Natural frequencies, shape (r,), rad/s.
        zn: Damping ratios, shape (r,).
        Phi: Mass-normalised mode shapes, shape (n_locs, r).
        frf_type: "compliance", "mobility" or "accelerance". Defaults to "accelerance".

    Returns:
        FRF matrix of shape (n_locs, n_locs, N); element [i, j] is the response at i to a
        force at j.
    """
    scaling = {
        "compliance": 1,
        "mobility": 1j * ws[:, None],
        "accelerance": -(ws[:, None] ** 2),
    }[frf_type]

    den = scaling / (wn**2 + 2j * zn * wn * ws[:, None] - ws[:, None] ** 2)
    A = Phi[:, None, :] * Phi[None, :, :]

    return A.astype(complex).dot(den.T)
