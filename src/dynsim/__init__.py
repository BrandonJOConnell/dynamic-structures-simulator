"""
dynsim: randomised mass-spring-damper towers and chains with ground-truth labels.

Build a baseline tower or chain, perturb it with randomised features (rescaled connections,
extra ground springs, rescaled masses, extra mass-spring-damper appendages), simulate it, and
draw it as a TikZ / SVG diagram.

>>> from dynsim import sample_system, render_svg
>>> rs = sample_system(42, "tower", 6)
>>> rs.describe()                      # ground truth: exactly what was changed
>>> rs.system.modal                    # natural frequencies (Hz), damping ratios, mode shapes
>>> render_svg(rs, "tower_42.svg")     # needs pdflatex and a PDF-to-SVG converter
"""

from importlib.metadata import PackageNotFoundError, version

from .randomise import (
    AddAppendage,
    AddGroundDamper,
    AddGroundSpring,
    AppendageSpec,
    Connection,
    GroundDamperSpec,
    GroundSpringSpec,
    RandomisedSystem,
    ScaleConnection,
    ScaleConnectionSpec,
    ScaleMass,
    ScaleMassSpec,
    build_system,
    sample_features,
    sample_system,
)
from .simulate import build_frf, simulate_discrete_ssm, van_loan_discretisation
from .systems import (
    ModalProperties,
    SystemMatrices,
    make_mdof_chain,
    make_mdof_tower,
    make_state_matrices,
    poles_to_modal,
)
from .tikz import render_batch, render_svg, to_standalone_tex, to_tikz, write_tex

try:
    __version__ = version("dynsim")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = [
    "AddAppendage",
    "AddGroundDamper",
    "AddGroundSpring",
    "AppendageSpec",
    "Connection",
    "GroundDamperSpec",
    "GroundSpringSpec",
    "ModalProperties",
    "RandomisedSystem",
    "ScaleConnection",
    "ScaleConnectionSpec",
    "ScaleMass",
    "ScaleMassSpec",
    "SystemMatrices",
    "build_frf",
    "build_system",
    "make_mdof_chain",
    "make_mdof_tower",
    "make_state_matrices",
    "poles_to_modal",
    "render_batch",
    "render_svg",
    "sample_features",
    "sample_system",
    "simulate_discrete_ssm",
    "to_standalone_tex",
    "to_tikz",
    "van_loan_discretisation",
    "write_tex",
]
