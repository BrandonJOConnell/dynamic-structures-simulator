import numpy as np
import pytest
import scipy.linalg as sla

from dynsim import (
    SystemMatrices,
    build_frf,
    build_system,
    make_mdof_chain,
    make_mdof_tower,
    make_state_matrices,
    sample_system,
)
from dynsim.systems import _align_phase

BUILDERS = {"tower": make_mdof_tower, "chain": make_mdof_chain}


def mac(a, b):
    return abs(a @ b) ** 2 / ((a @ a) * (b @ b))


@pytest.mark.parametrize("kind", ["tower", "chain"])
@pytest.mark.parametrize("ndof", [1, 2, 5, 8])
def test_build_system_baseline_matches_reference_builders(kind, ndof):
    built = build_system(kind, ndof).system
    ref = BUILDERS[kind](ndof)
    for a, b in zip(built, ref):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("kind", ["tower", "chain"])
def test_modal_matches_independent_eigensolve(kind):
    s = BUILDERS[kind](5)  # proportional damping: C = (c/k) K, so modes are real
    w2, V = sla.eigh(s.K, s.M)
    modal = s.modal

    np.testing.assert_allclose(modal.wn, np.sqrt(w2) / (2 * np.pi), rtol=1e-10)
    np.testing.assert_allclose(modal.zeta, (2 / 2500) * np.sqrt(w2) / 2, rtol=1e-8)
    for j in range(5):
        assert mac(modal.phi[:, j], V[:, j]) > 1 - 1e-10
    np.testing.assert_allclose(modal.phi.T @ s.M @ modal.phi, np.eye(5), atol=1e-10)
    assert np.all(np.diff(modal.wn) > 0)


def test_modal_complex_equals_real_modal_for_proportional_damping():
    s = make_mdof_tower(5)
    real, cplx = s.modal, s.modal_complex
    np.testing.assert_allclose(cplx.wn, real.wn)
    np.testing.assert_allclose(cplx.zeta, real.zeta)
    np.testing.assert_allclose(cplx.phi.real, real.phi, atol=1e-9)
    np.testing.assert_allclose(cplx.phi.imag, 0, atol=1e-9)


def test_modal_complex_satisfies_the_quadratic_eigenproblem():
    # (lambda^2 M + lambda C + K) phi = 0 for every mode of a strongly non-proportional system
    s = build_system(
        "tower",
        5,
        features=[],
    ).system
    s = SystemMatrices(s.M, s.K, s.C + np.diag([0.0, 30.0, 0.0, 0.0, 0.0]))  # one heavy local damper
    modal = s.modal_complex
    assert np.abs(modal.phi.imag).max() > 1e-3  # genuinely complex modes
    for j in range(5):
        wn = 2 * np.pi * modal.wn[j]
        lam = -modal.zeta[j] * wn + 1j * wn * np.sqrt(1 - modal.zeta[j] ** 2)
        residual = (lam**2 * s.M + lam * s.C + s.K) @ modal.phi[:, j]
        assert np.linalg.norm(residual) < 1e-8 * np.linalg.norm(s.K @ modal.phi[:, j])


@pytest.mark.parametrize("seed", range(50))
def test_modal_real_shapes_close_to_undamped_modes_for_sampled_systems(seed):
    s = sample_system(seed, "tower", 6).system
    _, V = sla.eigh(s.K, s.M)
    phi = s.modal.phi
    assert min(mac(phi[:, j], V[:, j]) for j in range(6)) > 0.95


def test_align_phase_removes_arbitrary_complex_scale():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(6)
    for _ in range(20):
        a = np.exp(1j * rng.uniform(0, 2 * np.pi)) * rng.uniform(0.1, 5)
        v = _align_phase(a * r)
        np.testing.assert_allclose(v.imag, 0, atol=1e-12)
        np.testing.assert_allclose(np.abs(v), np.abs(a) * np.abs(r), rtol=1e-12)


def test_modal_raises_for_overdamped_system():
    s = SystemMatrices(np.eye(1), np.eye(1), 10 * np.eye(1))
    with pytest.raises(ValueError, match="underdamped"):
        _ = s.modal


def test_state_matrices_shapes_and_options():
    s = make_mdof_tower(4)
    A, C = make_state_matrices(s)
    assert A.shape == (8, 8) and C.shape == (4, 8)
    assert make_state_matrices(s, "d", dofs=[0, 2])[1].shape == (2, 8)
    assert make_state_matrices(s, "velocity", dofs=1)[1].shape == (1, 8)  # int dofs; first letter used
    with pytest.raises(ValueError, match="observe"):
        make_state_matrices(s, "x")


@pytest.mark.parametrize("frf_type, scale", [
    ("compliance", lambda w: 1),
    ("mobility", lambda w: 1j * w),
    ("accelerance", lambda w: -(w**2)),
])
def test_build_frf_matches_direct_inversion(frf_type, scale):
    s = make_mdof_tower(4)
    modal = s.modal
    ws = 2 * np.pi * np.linspace(0.2, 12, 60)
    H = build_frf(ws, 2 * np.pi * modal.wn, modal.zeta, modal.phi, frf_type)
    assert H.shape == (4, 4, 60)
    for k, w in enumerate(ws):
        direct = scale(w) * np.linalg.inv(-(w**2) * s.M + 1j * w * s.C + s.K)
        np.testing.assert_allclose(H[:, :, k], direct, rtol=1e-8, atol=1e-14)
