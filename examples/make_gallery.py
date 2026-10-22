"""
Regenerate the SVGs shown in the README (docs/img/).

Needs pdflatex and a PDF-to-SVG converter (pdf2svg, pdftocairo or dvisvgm) on your PATH.

    uv run python examples/make_gallery.py
"""

from pathlib import Path

from dynsim import (
    AppendageSpec,
    GroundDamperSpec,
    GroundSpringSpec,
    ScaleConnectionSpec,
    ScaleMassSpec,
    build_system,
    render_svg,
    sample_system,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 111  # a seed whose draw shows every feature type clearly


def feature_types(kind: str) -> list:
    return [
        ScaleConnectionSpec(n=2, k_range=(0.5, 2.0), c_range=(0.5, 2.0)),  # k and c drawn independently
        GroundSpringSpec(n=1),  # a spring to ground, with no damper
        GroundDamperSpec(n=1),  # a damper to ground, with no spring
        ScaleMassSpec(n=(1, 2)),
        AppendageSpec(attach="top" if kind == "tower" else "random"),
    ]


systems = {
    "baseline_tower": build_system("tower", 5),
    "baseline_chain": build_system("chain", 5),
    "random_tower": sample_system(SEED, "tower", 6, feature_types=feature_types("tower")),
    "random_chain": sample_system(SEED, "chain", 5, feature_types=feature_types("chain")),
}

for name, rs in systems.items():
    path = render_svg(rs, OUT / f"{name}.svg")
    print(f"wrote {path}")
    for line in rs.describe():
        print(f"    {line}")
