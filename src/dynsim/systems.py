"""
System matrices, modal analysis, and the baseline tower / chain builders.

Conventions
-----------
* Equation of motion: ``M x'' + C x' + K x = f(t)``.
* ``modal`` reports natural frequencies in **Hz** (``poles_to_modal`` works in rad/s).
* Mode shapes are columns of ``phi`` and are mass-normalised: ``phi[:, j] @ M @ phi[:, j] == 1``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class ModalProperties(NamedTuple):
    """
    Modal properties of a system, ordered by increasing natural frequency.

    Attributes:
        wn: Natural frequencies (Hz), shape (ndof,).
        zeta: Damping ratios (-), shape (ndof,).
        phi: Mass-normalised mode shapes, shape (ndof, ndof); column j is mode j.
    """

    wn: NDArray
    zeta: NDArray
    phi: NDArray


class SystemMatrices(NamedTuple):
    """
    Mass, stiffness and damping matrices of a linear system.

    Attributes:
        M: Mass matrix, shape (ndof, ndof).
        K: Stiffness matrix, shape (ndof, ndof).
        C: Damping matrix, shape (ndof, ndof).
    """

    M: Array
    K: Array
    C: Array

    @property
    def ndof(self) -> int:
        """Number of degrees of freedom."""
        return self.M.shape[0]

    @property
    def modal(self) -> ModalProperties:
        """
        Modal properties with **real** mode shapes.

        Exact when damping is proportional. With non-proportional damping the true modes are
        complex; this returns the real part of each mode after rotating it to the phase that
        makes it as real as possible, which is a good approximation for light damping (use
        `modal_complex` for the complex modes themselves).

        Raises:
            ValueError: if the system has overdamped or rigid-body modes.
        """
        return _modal(self, complex_modes=False)

    @property
    def modal_complex(self) -> ModalProperties:
        """
        Modal properties with **complex** mode shapes (exact for any damping).

        Shapes are phase-aligned so they are as close to real as possible and scaled so that
        ``phi[:, j] @ M @ phi[:, j] == 1`` (the plain transpose, not the conjugate transpose).
        For proportional damping they are real and equal to `modal`.
        """
        return _modal(self, complex_modes=True)


def poles_to_modal(poles: NDArray) -> tuple[NDArray, NDArray]:
    """
    Convert complex poles to natural frequencies and damping ratios.

    Args:
        poles: Complex poles (continuous time), shape (P,).

    Returns:
        wn: Natural frequencies in **rad/s**, shape (P,).
        zn: Damping ratios, shape (P,).
    """
    wn = np.abs(poles)
    zn = -np.real(poles) / wn
    return wn, zn


def _align_phase(v: NDArray) -> NDArray:
    """Rotate a complex vector by the phase that makes its real part as large as possible."""
    return v * np.exp(-0.5j * np.angle(v @ v))


def _modal(system: SystemMatrices, complex_modes: bool) -> ModalProperties:
    n = system.ndof
    A, _ = make_state_matrices(system)
    lam, V = np.linalg.eig(A)

    # One pole of each complex-conjugate pair, in order of frequency.
    keep = np.flatnonzero(lam.imag > 0)
    if keep.size != n:
        raise ValueError(
            f"Expected {n} underdamped modes but found {keep.size}. Modal properties are only "
            "defined here for systems without overdamped or rigid-body modes."
        )
    keep = keep[np.argsort(np.abs(lam[keep]))]

    wn, zeta = poles_to_modal(lam[keep])

    # The top block of the eigenvector of A = [[0, I], [-M^-1 K, -M^-1 C]] is the mode shape
    # (the bottom block is lambda * shape). Eigenvectors carry an arbitrary complex scale, so
    # fix the phase explicitly rather than rely on the eigensolver's normalisation.
    phi = np.empty((n, n), dtype=complex if complex_modes else float)
    for j, col in enumerate(keep):
        v = _align_phase(V[:n, col])
        if not complex_modes:
            v = v.real
        v = v / np.sqrt(v @ system.M @ v)
        phi[:, j] = v if v.real[np.argmax(np.abs(v))] >= 0 else -v

    return ModalProperties(wn / (2 * np.pi), zeta, phi)


def make_mdof_tower(ndof: int, m: float = 5, k: float = 2500, c: float = 2) -> SystemMatrices:
    r"""
    System matrices for a simple tower where m, k, c are the same on every floor.

    Each storey has two spring-damper columns, so every connection has stiffness 2k and damping
    2c (see `dynsim.build_system` for per-floor masses and randomised variants).

     ____________________
    |_________m__________| (floor ndof)
       /  |        /  |
     k \ |_| c   k \ |_| c
       /  |        /  |
             .....
       \ |_|       \ |_|
     __/__|________/__|__
    |_________m__________| (floor 2)
       /  |        /  |
     k \ |_| c   k \ |_| c
     __/__|________/__|__
    |_________m__________| (floor 1)
       /  |        /  |
     k \ |_| c   k \ |_| c
    ___/__|________/__|___
    ////////////////////// (ground)

    Args:
        ndof: Number of degrees of freedom (floors).
        m: Mass of each floor (kg). Defaults to 5.
        k: Stiffness of each spring (N/m). Defaults to 2500.
        c: Damping of each damper (Ns/m). Defaults to 2.
    """
    M = m * np.eye(ndof)
    template = 4 * np.eye(ndof) - 2 * np.eye(ndof, k=1) - 2 * np.eye(ndof, k=-1)
    template[-1, -1] = 2  # the top floor is not connected to anything above it
    return SystemMatrices(M, k * template, c * template)


def make_mdof_chain(ndof: int, m: float = 5, k: float = 2500, c: float = 2) -> SystemMatrices:
    r"""
    System matrices for a chain of masses joined by springs and dampers, grounded at both ends.

    /|   k   _______    k   _______    k             k   _______    k  |/
    /|-\/\/-|       |-\/\/-|       |-\/\/-        -\/\/-|       |-\/\/-|/
    /|      |   m   |      |   m   |       ......       |   m   |      |/
    /|-|[---|_______|-|[---|_______|-|[---        -|[---|_______|-|[---|/
    /|   c              c              c             c              c  |/

    Args:
        ndof: Number of degrees of freedom (masses).
        m: Mass of each mass (kg). Defaults to 5.
        k: Spring stiffness (N/m). Defaults to 2500.
        c: Damper coefficient (Ns/m). Defaults to 2.
    """
    M = m * np.eye(ndof)
    template = 2 * np.eye(ndof) - np.eye(ndof, k=1) - np.eye(ndof, k=-1)
    return SystemMatrices(M, k * template, c * template)


def make_state_matrices(
    system: SystemMatrices, observe: str = "a", dofs: int | list[int] | NDArray | None = None
) -> tuple[Array, Array]:
    """
    Continuous-time state-space matrices ``x' = A x``, ``y = C x`` with ``x = [displacement; velocity]``.

    Note that the output matrix has **no feedthrough term**. For accelerations the full
    observation equation is ``y = C x + M^-1 B u`` for an input entering as ``B u``; the missing
    term does not change the poles.

    Args:
        system: System matrices M, K, C.
        observe: What to observe: "d" (displacement), "v" (velocity) or "a" (acceleration).
            Only the first letter is used. Defaults to "a".
        dofs: Which dofs to observe (int or sequence of ints). Defaults to all.

    Returns:
        A: State matrix, shape (2 ndof, 2 ndof).
        C: Output matrix, shape (len(dofs), 2 ndof).

    Raises:
        ValueError: if `observe` is not one of "d", "v", "a".
    """
    ndof = system.ndof
    dofs = np.arange(ndof) if dofs is None else np.atleast_1d(dofs)

    MIK = np.linalg.solve(system.M, system.K)
    MIC = np.linalg.solve(system.M, system.C)

    A = np.block([[np.zeros((ndof, ndof)), np.eye(ndof)], [-MIK, -MIC]])

    match observe[:1]:
        case "d":
            C = np.hstack((np.eye(ndof), np.zeros((ndof, ndof))))
        case "v":
            C = np.hstack((np.zeros((ndof, ndof)), np.eye(ndof)))
        case "a":
            C = np.hstack((-MIK, -MIC))
        case _:
            raise ValueError(f"observe must start with 'd', 'v' or 'a', got {observe!r}")

    return A, C[dofs]
