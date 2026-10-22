"""
Randomised MDOF towers and chains (linear).

A system is a baseline (tower or chain) plus a list of "features" that perturb it:

    ScaleConnection  - multiply the stiffness and/or damping of an existing connection
    AddGroundSpring  - add an extra spring (and optional damper) from a floor to ground
    AddGroundDamper  - add an extra damper (no spring) from a floor to ground
    ScaleMass        - multiply the lumped mass of one floor
    AddAppendage     - attach an extra mass (new dof) to a floor through a spring and damper,
                       e.g. a small mass on top of a tower (system size grows by one)

K, C and M are assembled from connection elements and lumped masses, so every feature is just
a change to those. The baseline reproduces make_mdof_tower / make_mdof_chain exactly.

Choosing which feature types are randomised
-------------------------------------------
Each feature type has a matching sampler "spec" that says how to draw features of that type.
Pass the ones you want via `feature_types`, either by name (default settings) or as spec
objects (custom settings):

>>> sample_system(42, "tower", 6)                                   # all types, defaults
>>> sample_system(42, "tower", 6, feature_types=["scale", "mass"])  # no ground springs
>>> sample_system(42, "chain", 8, feature_types=[
...     ScaleConnectionSpec(n=(1, 1), factor_range=(0.25, 0.5)),    # exactly one softened link
...     ScaleMassSpec(n=(0, 3)),
... ])
>>> sample_system(42, "tower", 6, feature_types=[])                 # unmodified baseline

How many of each type
---------------------
Every spec takes `n`, either an inclusive (min, max) range or a single int for an exact count:

>>> ScaleConnectionSpec(n=(1, 3))   # one to three changes
>>> ScaleConnectionSpec(n=1)        # exactly one   (same as n=(1, 1))
>>> ScaleConnectionSpec(n=0)        # none          (same as n=(0, 0))

Counts are drawn per type and capped at what exists (distinct connections / floors).

Stiffness and damping
---------------------
Damping is never changed unless you ask for it. Each spec changes only what you give it:

>>> ScaleConnectionSpec(k_range=(0.5, 2))                      # stiffness only (also the default)
>>> ScaleConnectionSpec(c_range=(0.5, 2))                      # damping only
>>> ScaleConnectionSpec(k_range=(0.5, 2), c_range=(0.5, 2))    # both, drawn independently
>>> ScaleConnectionSpec(k_range=(0.5, 2), couple_damping=True) # damping gets the SAME factor
>>> GroundSpringSpec()                                         # spring only
>>> GroundSpringSpec(couple_damping=True)                      # spring plus a damper (baseline c/k)
>>> GroundDamperSpec()                                         # damper only

`couple_damping=True` is the one way damping silently follows stiffness, and it is opt-in. It
cannot be combined with `c_range` (that raises a ValueError). Appendages are a mass-spring-
damper, so `AppendageSpec` includes a damper by default (`couple_damping=False` removes it).

Targeting where a change happens
--------------------------------
Give a spec `targets` to restrict it to chosen connections (ScaleConnectionSpec) or floors
(GroundSpringSpec, ScaleMassSpec; AppendageSpec uses `attach`). Magnitudes are still random
within the spec's range, so pin them with e.g. factor_range=(2, 2). With `targets` and no `n`,
every target is changed; give `n` to pick a random subset of the targets instead.

>>> ScaleConnectionSpec(targets=3, factor_range=(0.3, 0.6))   # soften connection 3, by 0.3-0.6x
>>> ScaleMassSpec(targets=[2, 4], factor_range=(2, 2))        # double the mass of floors 2 and 4
>>> GroundSpringSpec(targets=[0, 5], n=1)                     # one ground spring, on floor 0 or 5

Extra mass-spring-damper on top
-------------------------------
>>> sample_system(42, "tower", 6, feature_types=["appendage"])      # one, on the top floor
>>> sample_system(42, "chain", 6, feature_types=[AppendageSpec(attach="random", n=(1, 2))])
>>> build_system("tower", 6, features=[AddAppendage(attach=5, m=1.0, k=800, c=0.6)])

"top" means the last floor (index ndof-1: the top of a tower, the end mass of a chain).
Appendages add dofs, so rs.system has ndof + (number of appendages) dofs while rs.ndof stays
the baseline count. Appendage dofs are numbered after the baseline ones.

Names: "scale", "ground", "mass", "ground_damper", "appendage". The defaults are "scale",
"ground" and "mass".
Anything with a `sample(rng, ctx) -> list[Feature]` method also works, so you can add your
own types.

Other usage
-----------
>>> rs = sample_system(42, "tower", 6)
>>> rs.system.K          # perturbed stiffness matrix
>>> rs.describe()        # human-readable record of what was changed
>>> rs.features          # the exact features applied (ground truth)

Note: ground springs and local stiffness changes make damping non-proportional, so the true
modes are complex. `SystemMatrices.modal` gives a real approximation (fine for light damping);
`SystemMatrices.modal_complex` gives the complex modes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .systems import SystemMatrices

Kind = Literal["tower", "chain"]


# --------------------------------------------------------------------------------------
# Elements and assembly
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Connection:
    """Spring + damper between floor `i` and floor `j` (j=None means ground)."""

    i: int
    j: int | None
    k: float
    c: float

    def describe(self) -> str:
        end = "ground" if self.j is None else f"floor {self.j}"
        return f"floor {self.i} <-> {end}: k={self.k:.4g}, c={self.c:.4g}"


def assemble(masses: np.ndarray, connections: Sequence[Connection]) -> SystemMatrices:
    """Assemble M, K, C from lumped masses and connection elements."""
    ndof = len(masses)
    K = np.zeros((ndof, ndof))
    C = np.zeros((ndof, ndof))
    for e in connections:
        L = np.zeros(ndof)  # picks out the relative displacement across the element
        L[e.i] = 1.0
        if e.j is not None:
            L[e.j] = -1.0
        LL = np.outer(L, L)
        K += e.k * LL
        C += e.c * LL
    return SystemMatrices(np.diag(masses), K, C)


def baseline_connections(kind: Kind, ndof: int, k: float, c: float) -> list[Connection]:
    """
    Baseline connection list. Order is fixed so ScaleConnection indices are meaningful:

        index 0                 : ground <-> floor 0
        index 1 .. ndof-1       : floor (n-1) <-> floor n
        index ndof (chain only) : floor ndof-1 <-> ground

    Tower: every connection has stiffness 2k and damping 2c (two springs per storey in
    make_mdof_tower), and only the base is grounded.
    Chain: every connection has k, c, and both ends are grounded.
    """
    if kind == "tower":
        kk, cc = 2 * k, 2 * c
    elif kind == "chain":
        kk, cc = k, c
    else:
        raise ValueError(f"kind must be 'tower' or 'chain', got {kind!r}")

    conns = [Connection(0, None, kk, cc)]
    conns += [Connection(n, n - 1, kk, cc) for n in range(1, ndof)]
    if kind == "chain":
        conns.append(Connection(ndof - 1, None, kk, cc))
    return conns


# --------------------------------------------------------------------------------------
# Features (what gets applied to a system)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScaleConnection:
    """Scale an existing baseline connection (index into baseline_connections)."""

    index: int
    k_factor: float = 1.0
    c_factor: float = 1.0

    def apply(self, conns: list[Connection], masses: np.ndarray) -> None:
        e = conns[self.index]
        conns[self.index] = replace(e, k=e.k * self.k_factor, c=e.c * self.c_factor)

    def describe(self) -> str:
        parts = []
        if self.k_factor != 1.0:
            parts.append(f"k x{self.k_factor:.3g}")
        if self.c_factor != 1.0:
            parts.append(f"c x{self.c_factor:.3g}")
        return f"connection {self.index}: {', '.join(parts) or 'unchanged'}"


@dataclass(frozen=True)
class AddGroundSpring:
    """Extra spring (and optional damper) from floor `dof` to ground."""

    dof: int
    k: float
    c: float = 0.0

    def apply(self, conns: list[Connection], masses: np.ndarray) -> None:
        conns.append(Connection(self.dof, None, self.k, self.c))

    def describe(self) -> str:
        return f"ground spring at floor {self.dof}: k={self.k:.4g}, c={self.c:.4g}"


@dataclass(frozen=True)
class AddGroundDamper:
    """Extra damper `c` (no spring) from floor `dof` to ground."""

    dof: int
    c: float

    def apply(self, conns: list[Connection], masses: np.ndarray) -> None:
        conns.append(Connection(self.dof, None, 0.0, self.c))

    def describe(self) -> str:
        return f"ground damper at floor {self.dof}: c={self.c:.4g}"


@dataclass(frozen=True)
class ScaleMass:
    """Multiply the lumped mass of floor `dof`."""

    dof: int
    factor: float

    def apply(self, conns: list[Connection], masses: np.ndarray) -> None:
        masses[self.dof] *= self.factor

    def describe(self) -> str:
        return f"mass at floor {self.dof}: x{self.factor:.3g}"


@dataclass(frozen=True)
class AddAppendage:
    """
    Extra mass `m` attached to floor `attach` by a spring `k` and damper `c`.

    The new mass is a new dof, numbered after all dofs that exist when it is applied (so the
    first appendage gets index ndof, the next ndof + 1, ...).
    """

    attach: int
    m: float
    k: float
    c: float = 0.0

    def apply(self, conns: list[Connection], masses: list[float]) -> None:
        if not 0 <= self.attach < len(masses):
            raise IndexError(f"AddAppendage attach {self.attach} outside 0..{len(masses) - 1}")
        new = len(masses)
        masses.append(float(self.m))
        conns.append(Connection(new, self.attach, self.k, self.c))

    def describe(self) -> str:
        return f"appendage on floor {self.attach}: m={self.m:.4g}, k={self.k:.4g}, c={self.c:.4g}"


Feature = ScaleConnection | AddGroundSpring | AddGroundDamper | ScaleMass | AddAppendage


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RandomisedSystem:
    """
    A built system together with the ground-truth record of how it was made.

    `ndof` is the number of baseline floors. If appendages were added, `system.ndof` is larger.
    """

    system: SystemMatrices
    kind: Kind
    ndof: int
    features: tuple[Feature, ...]
    connections: tuple[Connection, ...]

    def describe(self) -> list[str]:
        """One line per feature (empty list = unmodified baseline)."""
        return [f.describe() for f in self.features]


def build_system(
    kind: Kind,
    ndof: int,
    m: float | np.ndarray = 5,
    k: float = 2500,
    c: float = 2,
    features: Sequence[Feature] = (),
) -> RandomisedSystem:
    """Build a baseline tower/chain and apply the given features, in order."""
    masses = np.broadcast_to(np.asarray(m, dtype=float), (ndof,)).tolist()  # list: features may append
    conns = baseline_connections(kind, ndof, k, c)
    n_base = len(conns)

    for f in features:
        if isinstance(f, ScaleConnection) and not 0 <= f.index < n_base:
            raise IndexError(f"ScaleConnection index {f.index} outside baseline 0..{n_base - 1}")
        if isinstance(f, (AddGroundSpring, AddGroundDamper, ScaleMass)) and not 0 <= f.dof < ndof:
            raise IndexError(f"{type(f).__name__} dof {f.dof} outside 0..{ndof - 1}")
        f.apply(conns, masses)

    return RandomisedSystem(
        system=assemble(masses, conns),
        kind=kind,
        ndof=ndof,
        features=tuple(features),
        connections=tuple(conns),
    )


# --------------------------------------------------------------------------------------
# Sampler specs (how features of each type are drawn)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingContext:
    """Baseline information handed to every spec's `sample` method."""

    kind: Kind
    ndof: int
    m: np.ndarray  # baseline masses, shape (ndof,)
    k: float
    c: float

    @property
    def n_connections(self) -> int:
        return len(baseline_connections(self.kind, self.ndof, self.k, self.c))

    @property
    def k_connection(self) -> float:
        """Baseline stiffness of a single connection."""
        return 2 * self.k if self.kind == "tower" else self.k

    @property
    def c_connection(self) -> float:
        """Baseline damping of a single connection."""
        return 2 * self.c if self.kind == "tower" else self.c

    @property
    def c_over_k(self) -> float:
        """Baseline damping-to-stiffness ratio (same for tower and chain)."""
        return self.c / self.k


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _as_range(n) -> tuple[int, int]:
    """Normalise `n`: an int means an exact count, a pair means an inclusive (min, max) range."""
    if isinstance(n, (int, np.integer)) and not isinstance(n, bool):
        lo = hi = int(n)
    else:
        try:
            lo, hi = n
        except (TypeError, ValueError):
            raise ValueError(f"n must be an int or a (min, max) pair, got {n!r}") from None
    if lo < 0 or hi < lo:
        raise ValueError(f"n must satisfy 0 <= min <= max, got {n!r}")
    return int(lo), int(hi)


def _count(rng: np.random.Generator, n, available: int) -> int:
    """Draw an integer from `n` (int or inclusive range), capped at what is available."""
    lo, hi = _as_range(n)
    return min(int(rng.integers(lo, hi + 1)), available)


def _check_range(name: str, r) -> None:
    """Raise ValueError unless `r` is a (lo, hi) pair with 0 < lo <= hi."""
    try:
        lo, hi = r
        ok = 0 < lo <= hi
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{name} must be a (lo, hi) pair with 0 < lo <= hi, got {r!r}")


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _resolve_targets(targets, size: int, what: str) -> list[int] | None:
    """Validate `targets` (None, an int, or a sequence of ints) against the indices 0..size-1."""
    if targets is None:
        return None
    if isinstance(targets, str):
        raise TypeError(f"targets must be an int or a sequence of ints, got {targets!r}")
    try:
        items = [targets] if _is_int(targets) else list(targets)
    except TypeError:
        raise TypeError(f"targets must be an int or a sequence of ints, got {targets!r}") from None
    if not all(_is_int(t) for t in items):
        raise TypeError(f"targets must be integers, got {targets!r}")
    cands = list(dict.fromkeys(int(t) for t in items))  # drop duplicates, keep order
    if not cands:
        raise ValueError("targets is empty; use None to allow any location")
    bad = [t for t in cands if not 0 <= t < size]
    if bad:
        raise IndexError(f"{what} targets {bad} outside 0..{size - 1}")
    return cands


def _pick(rng: np.random.Generator, n, targets, size: int, what: str) -> list[int]:
    """
    Choose the (distinct) indices a spec acts on.

    Without `targets`: draw a count from `n` (default (0, 2)) and pick that many of 0..size-1.
    With `targets`: pick from those only; if `n` is None, use all of them.
    """
    cands = _resolve_targets(targets, size, what)
    if cands is None:
        k = _count(rng, (0, 2) if n is None else n, size)
        return [int(i) for i in rng.choice(size, size=k, replace=False)]
    k = len(cands) if n is None else _count(rng, n, len(cands))
    return [cands[int(i)] for i in rng.choice(len(cands), size=k, replace=False)]


@dataclass(frozen=True)
class ScaleConnectionSpec:
    """
    Rescale the stiffness and/or damping of some baseline connections.

    Only what you give is changed: `k_range` rescales stiffness, `c_range` rescales damping,
    and giving both draws the two factors independently. If you give neither, stiffness is
    rescaled by (0.5, 2.0). Damping never follows stiffness unless you set `couple_damping=True`.

    Args:
        n: exact count (int) or inclusive (min, max) range for how many connections to rescale
            (distinct connections). Default: (0, 2), or every target if `targets` is given.
        k_range: log-uniform range for the stiffness multiplier, or None to leave stiffness
            alone. (0.5, 2.0) is symmetric in log space, so softening and stiffening are
            equally likely.
        c_range: log-uniform range for the damping multiplier, or None to leave damping alone.
        couple_damping: if True, damping gets the same factor as stiffness (the local
            damping-to-stiffness ratio is preserved). Cannot be combined with `c_range`.
        targets: restrict the change to these connection indices (an int or a sequence; see
            `baseline_connections` for the numbering). None means any connection.
    """

    n: tuple[int, int] | int | None = None
    k_range: tuple[float, float] | None = None
    c_range: tuple[float, float] | None = None
    couple_damping: bool = False
    targets: int | Sequence[int] | None = None

    def __post_init__(self):
        for name in ("k_range", "c_range"):
            if getattr(self, name) is not None:
                _check_range(name, getattr(self, name))
        if self.couple_damping and self.c_range is not None:
            raise ValueError(
                "couple_damping=True gives damping the same factor as stiffness, so it cannot be "
                "combined with c_range. Use one or the other."
            )

    def _ranges(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if self.k_range is None and self.c_range is None:
            return (0.5, 2.0), None
        return self.k_range, self.c_range

    def sample(self, rng: np.random.Generator, ctx: SamplingContext) -> list[Feature]:
        k_rng, c_rng = self._ranges()
        out: list[Feature] = []
        for idx in _pick(rng, self.n, self.targets, ctx.n_connections, "connection"):
            k_fac = _log_uniform(rng, *k_rng) if k_rng else 1.0
            if self.couple_damping:
                c_fac = k_fac
            else:
                c_fac = _log_uniform(rng, *c_rng) if c_rng else 1.0
            out.append(ScaleConnection(idx, k_fac, c_fac))
        return out


@dataclass(frozen=True)
class GroundSpringSpec:
    """
    Add extra ground springs (spring only, unless `couple_damping=True`).

    Args:
        n: exact count (int) or inclusive (min, max) range for how many ground springs to add
            (on distinct floors). Default: (0, 2), or every target if `targets` is given.
        k_range: log-uniform range for the spring stiffness, as a multiple of the baseline
            connection stiffness.
        couple_damping: if True, each spring also gets a damper with the baseline c/k ratio.
            If False (the default), the ground spring has no damper. For an independent
            damper use `GroundDamperSpec`.
        targets: restrict the springs to these floors (an int or a sequence of floor indices).
            None means any floor.
    """

    n: tuple[int, int] | int | None = None
    k_range: tuple[float, float] = (0.1, 1.0)
    couple_damping: bool = False
    targets: int | Sequence[int] | None = None

    def __post_init__(self):
        _check_range("k_range", self.k_range)

    def sample(self, rng: np.random.Generator, ctx: SamplingContext) -> list[Feature]:
        out: list[Feature] = []
        for dof in _pick(rng, self.n, self.targets, ctx.ndof, "floor"):
            kg = ctx.k_connection * _log_uniform(rng, *self.k_range)
            out.append(AddGroundSpring(dof, kg, kg * ctx.c_over_k if self.couple_damping else 0.0))
        return out


@dataclass(frozen=True)
class GroundDamperSpec:
    """
    Add extra ground dampers (no spring).

    Args:
        n: exact count (int) or inclusive (min, max) range for how many ground dampers to add
            (on distinct floors). Default: (0, 2), or every target if `targets` is given.
        c_range: log-uniform range for the damping coefficient, as a multiple of the baseline
            connection damping.
        targets: restrict the dampers to these floors (an int or a sequence of floor indices).
            None means any floor.
    """

    n: tuple[int, int] | int | None = None
    c_range: tuple[float, float] = (0.1, 1.0)
    targets: int | Sequence[int] | None = None

    def __post_init__(self):
        _check_range("c_range", self.c_range)

    def sample(self, rng: np.random.Generator, ctx: SamplingContext) -> list[Feature]:
        out: list[Feature] = []
        for dof in _pick(rng, self.n, self.targets, ctx.ndof, "floor"):
            out.append(AddGroundDamper(dof, ctx.c_connection * _log_uniform(rng, *self.c_range)))
        return out


@dataclass(frozen=True)
class ScaleMassSpec:
    """
    Rescale the mass of some floors.

    Args:
        n: exact count (int) or inclusive (min, max) range for how many floors to change
            (distinct floors). Default: (0, 2), or every target if `targets` is given.
        factor_range: log-uniform range for the mass multiplier (symmetric by default).
        targets: restrict the change to these floors (an int or a sequence of floor indices).
            None means any floor.
    """

    n: tuple[int, int] | int | None = None
    factor_range: tuple[float, float] = (0.5, 2.0)
    targets: int | Sequence[int] | None = None

    def __post_init__(self):
        _check_range("factor_range", self.factor_range)

    def sample(self, rng: np.random.Generator, ctx: SamplingContext) -> list[Feature]:
        out: list[Feature] = []
        for dof in _pick(rng, self.n, self.targets, ctx.ndof, "floor"):
            out.append(ScaleMass(dof, _log_uniform(rng, *self.factor_range)))
        return out


@dataclass(frozen=True)
class AppendageSpec:
    """
    Attach extra masses (each a new dof) to floors through a spring and damper.

    Unlike the other specs, the default is exactly one appendage (n=1), on the top floor:
    if you ask for "appendage" you normally want one.

    Args:
        n: exact count (int) or inclusive (min, max) range, on distinct floors. Capped at the
            number of candidate floors, so attach="top" gives at most one.
        attach: where to attach. "top" = last floor (index ndof-1: top of a tower, end mass of
            a chain); "random" = any baseline floor; an int = that floor; a sequence of ints =
            pick from those floors.
        m_range: log-uniform range for the appendage mass, as a multiple of the baseline mass
            of the floor it attaches to.
        k_range: log-uniform range for the spring stiffness, as a multiple of the baseline
            connection stiffness.
        couple_damping: if True (the default), the appendage has a damper with the baseline
            c/k ratio, since an appendage is a mass-spring-damper. If False, no damper.
    """

    n: tuple[int, int] | int = 1
    attach: str | int | Sequence[int] = "top"
    m_range: tuple[float, float] = (0.1, 1.0)
    k_range: tuple[float, float] = (0.1, 1.0)
    couple_damping: bool = True

    def __post_init__(self):
        _check_range("m_range", self.m_range)
        _check_range("k_range", self.k_range)

    def _candidates(self, ctx: SamplingContext) -> list[int]:
        a = self.attach
        if isinstance(a, str):
            if a == "top":
                return [ctx.ndof - 1]
            if a == "random":
                return list(range(ctx.ndof))
            raise ValueError(f"attach must be 'top', 'random', an int or a sequence of ints, got {a!r}")
        cands = [int(a)] if isinstance(a, (int, np.integer)) else [int(x) for x in a]
        bad = [x for x in cands if not 0 <= x < ctx.ndof]
        if bad:
            raise IndexError(f"attach floors {bad} outside 0..{ctx.ndof - 1}")
        return cands

    def sample(self, rng: np.random.Generator, ctx: SamplingContext) -> list[Feature]:
        out: list[Feature] = []
        cands = self._candidates(ctx)
        n = _count(rng, self.n, len(cands))
        for idx in rng.choice(len(cands), size=n, replace=False):
            floor = cands[int(idx)]
            m = float(ctx.m[floor] * _log_uniform(rng, *self.m_range))
            k = ctx.k_connection * _log_uniform(rng, *self.k_range)
            out.append(AddAppendage(floor, m, k, k * ctx.c_over_k if self.couple_damping else 0.0))
        return out


# Names accepted in `feature_types`; each maps to a spec with default settings.
FEATURE_SPECS = {
    "scale": ScaleConnectionSpec,
    "ground": GroundSpringSpec,
    "ground_damper": GroundDamperSpec,
    "mass": ScaleMassSpec,
    "appendage": AppendageSpec,
}

DEFAULT_FEATURE_TYPES = ("scale", "ground", "mass")


def _resolve_spec(spec):
    if isinstance(spec, str):
        if spec not in FEATURE_SPECS:
            raise ValueError(f"Unknown feature type {spec!r}. Options: {sorted(FEATURE_SPECS)}")
        return FEATURE_SPECS[spec]()
    if not hasattr(spec, "sample"):
        raise TypeError(f"feature type must be a name or an object with .sample(rng, ctx), got {spec!r}")
    return spec


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def sample_features(
    rng: np.random.Generator,
    kind: Kind,
    ndof: int,
    m: float | np.ndarray = 5,
    k: float = 2500,
    c: float = 2,
    feature_types: Sequence = DEFAULT_FEATURE_TYPES,
) -> list[Feature]:
    """
    Draw a random feature list. Specs are sampled in the order given, and their features are
    applied in that order. See the module docstring for how to choose `feature_types`.
    """
    masses = np.broadcast_to(np.asarray(m, dtype=float), (ndof,)).copy()
    ctx = SamplingContext(kind, ndof, masses, k, c)
    features: list[Feature] = []
    for spec in feature_types:
        features += _resolve_spec(spec).sample(rng, ctx)
    return features


def sample_system(
    rng: int | np.random.Generator | None,
    kind: Kind,
    ndof: int,
    m: float | np.ndarray = 5,
    k: float = 2500,
    c: float = 2,
    feature_types: Sequence = DEFAULT_FEATURE_TYPES,
) -> RandomisedSystem:
    """Sample features and build the system. `rng` can be a seed or a Generator."""
    rng = np.random.default_rng(rng)  # returns Generators unchanged
    feats = sample_features(rng, kind, ndof, m=m, k=k, c=c, feature_types=feature_types)
    return build_system(kind, ndof, m=m, k=k, c=c, features=feats)
