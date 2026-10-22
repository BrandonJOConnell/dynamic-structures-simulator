import numpy as np
from scipy.linalg import expm
from scipy.signal import find_peaks, welch

from dynsim import make_mdof_chain, make_state_matrices, simulate_discrete_ssm, van_loan_discretisation


def test_van_loan_matches_analytic_scalar_solution():
    # dx = -a x dt + b u dt + dβ (spectral density q):  A = e^{-a dt},  Q = q (1 - e^{-2 a dt}) / (2a)
    a, b, q, dt = 3.0, 2.0, 0.7, 0.05
    A, Q, Bd = van_loan_discretisation(np.array([[-a]]), q, np.array([1.0]), dt, B=np.array([[b]]))
    np.testing.assert_allclose(A, np.exp(-a * dt))
    np.testing.assert_allclose(Q, q * (1 - np.exp(-2 * a * dt)) / (2 * a))
    np.testing.assert_allclose(Bd, b * (1 - np.exp(-a * dt)) / a)


def test_van_loan_always_returns_three_values_and_psd_noise_covariance():
    s = make_mdof_chain(3)
    F, _ = make_state_matrices(s)
    L = np.vstack([np.zeros((3, 3)), np.linalg.inv(s.M)])
    A, Q, Bd = van_loan_discretisation(F, 1.0, L, 0.01)
    assert Bd is None
    np.testing.assert_allclose(A, expm(F * 0.01))
    np.testing.assert_allclose(Q, Q.T, atol=1e-12)
    assert np.linalg.eigvalsh(Q).min() > -1e-12


def test_simulate_is_reproducible_and_aligned():
    A, C = np.array([[0.9]]), np.array([[2.0]])
    Q, R = np.array([[0.01]]), np.array([[1e-20]])
    x1, y1 = simulate_discrete_ssm(A, C, Q, R, np.array([1.0]), T=50, rng=3)
    x2, y2 = simulate_discrete_ssm(A, C, Q, R, np.array([1.0]), T=50, rng=3)
    x3, _ = simulate_discrete_ssm(A, C, Q, R, np.array([1.0]), T=50, rng=4)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.allclose(x1, x3)
    assert x1.shape == y1.shape == (1, 50) and x1[0, 0] == 1.0
    np.testing.assert_allclose(y1, C @ x1, atol=1e-8)  # states and observations share a time index


def test_simulate_deterministic_part_and_inputs():
    A, C, R = np.array([[0.5]]), np.array([[1.0]]), np.array([[1e-20]])
    Q = np.zeros((1, 1))
    x, _ = simulate_discrete_ssm(A, C, Q, R, np.array([1.0]), T=6, rng=0)
    np.testing.assert_allclose(x[0], 0.5 ** np.arange(6), atol=1e-5)  # jitter adds ~1e-6 noise

    u = np.ones((1, 6))
    x, _ = simulate_discrete_ssm(A, C, Q, R, np.array([0.0]), B=np.array([[1.0]]), u=u, T=6, rng=0)
    expected = [0.0]
    for _ in range(5):
        expected.append(0.5 * expected[-1] + 1.0)
    np.testing.assert_allclose(x[0], expected, atol=1e-5)


def test_simulated_spectrum_peaks_at_true_natural_frequencies():
    s = make_mdof_chain(5)
    n, fs = s.ndof, 40.0
    F, C = make_state_matrices(s, "a")
    L = np.vstack([np.zeros((n, n)), np.linalg.inv(s.M)])  # white force noise on every dof
    A, Q, _ = van_loan_discretisation(F, 1.0, L, 1 / fs)

    _, y = simulate_discrete_ssm(A, C, Q, 1e-8 * np.eye(n), np.zeros(2 * n), T=2**17, rng=0)
    f, pxx = welch(y, fs=fs, nperseg=2**12)
    peaks, _ = find_peaks(pxx.mean(axis=0), prominence=pxx.mean(axis=0).max() * 1e-3)

    for wn in s.modal.wn:
        assert np.min(np.abs(f[peaks] - wn)) < 0.03
