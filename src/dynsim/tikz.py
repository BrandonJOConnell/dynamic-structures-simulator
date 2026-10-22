"""
TikZ mass-spring-damper diagrams (and SVG export) for randomised MDOF systems.

Works on the `RandomisedSystem` objects from `dynsim.randomise`:

    towers are drawn vertically (floor 0 at the bottom, ground below, extra ground springs
    run sideways to a wall on the right),
    chains are drawn horizontally (walls at both ends, extra ground springs run down to a
    ground line).

What the picture shows
----------------------
Every connection is a spring (coil) plus a damper (dashpot); an element with no stiffness
(a ground damper) or no damping (a ground spring without a damper) is drawn without the
missing symbol. With labels="values" each one is annotated with its effective k and c, and
each mass with its value. Colours mark what the
sampler changed relative to the baseline:

    red     stiffness / damping increased        blue   decreased
    green   extra ground link added (a line), or an extra mass-spring-damper (appendage) added
            (a green-outlined block)
    masses: orange fill = heavier, pale orange = lighter (block size also changes a little)

An appendage (AddAppendage) is drawn as a smaller green-outlined block: above the top floor of
a tower, to the left of any other floor of a tower, and above the attached mass in a chain.

Floors are numbered from `index_base` (default 0, matching the matrix indices and
`RandomisedSystem.describe()`).

Usage
-----
>>> from dynsim import sample_system
>>> from dynsim import to_tikz, to_standalone_tex, render_svg, render_batch
>>> rs = sample_system(42, "tower", 6)
>>> print(to_tikz(rs))                       # tikzpicture to paste into a document
>>> render_svg(rs, "figs/tower_42.svg")      # needs pdflatex + an SVG converter (see below)
>>> render_batch([sample_system(s, "chain", 5) for s in range(10)], "figs", stem="chain")

To paste `to_tikz` output into your own document you need, in the preamble:

    \\usepackage{tikz}
    \\usetikzlibrary{decorations.pathmorphing}

render_svg compiles with pdflatex, then converts the PDF with the first of pdf2svg,
pdftocairo or dvisvgm that it finds on your PATH.

Notes
-----
* Extra ground springs on the same floor are parallel springs, so they are drawn (and
  labelled) as one combined spring and damper. Repeated scalings of the same connection or
  mass are combined into one factor.
* Only features from dynsim.randomise are decoded for colouring. Values in the labels
  always come from the final matrices.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np

from .randomise import RandomisedSystem, ScaleConnection, ScaleMass

Labels = Literal["values", "none"]

TIKZ_LIBS = r"\usetikzlibrary{decorations.pathmorphing}"

_COLOURS = {
    "msdup": (200, 60, 40),
    "msddown": (40, 100, 200),
    "msdadded": (30, 140, 70),
    "msdmadded": (208, 236, 218),
    "msdmbase": (236, 236, 236),
    "msdmheavy": (244, 178, 106),
    "msdmlight": (252, 228, 200),
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.3g}"
    return f"{x:.2g}"


def _changed(factor: float) -> bool:
    return abs(math.log(factor)) > 1e-9


def _status_colour(factor: float) -> str:
    if not _changed(factor):
        return "black"
    return "msdup" if factor > 1 else "msddown"


def _decode(rs: RandomisedSystem):
    """Recover scaling factors and merged extra ground springs from the system record."""
    n_base = rs.ndof + (1 if rs.kind == "chain" else 0)
    kf: dict[int, float] = {}
    cf: dict[int, float] = {}
    mf: dict[int, float] = {}
    for f in rs.features:
        if isinstance(f, ScaleConnection):
            kf[f.index] = kf.get(f.index, 1.0) * f.k_factor
            cf[f.index] = cf.get(f.index, 1.0) * f.c_factor
        elif isinstance(f, ScaleMass):
            mf[f.dof] = mf.get(f.dof, 1.0) * f.factor

    n = rs.ndof
    base = list(rs.connections[:n_base])
    extra: dict[int, list[float]] = {}  # floor -> [k, c]; parallel ground springs add
    app: dict[int, tuple[int, float, float]] = {}  # attached floor -> (appendage dof, k, c)
    for e in rs.connections[n_base:]:
        if e.j is None and e.i < n:
            acc = extra.setdefault(e.i, [0.0, 0.0])
            acc[0] += e.k
            acc[1] += e.c
        elif e.j is not None and e.i >= n > e.j:
            if e.j in app:
                raise ValueError("Renderer supports at most one appendage per floor.")
            app[e.j] = (e.i, e.k, e.c)
        else:
            raise ValueError(
                "Renderer only supports extra ground springs on baseline floors and "
                "appendages attached to baseline floors."
            )
    return base, kf, cf, mf, extra, app


# --------------------------------------------------------------------------------------
# TikZ generation
# --------------------------------------------------------------------------------------
def to_tikz(
    rs: RandomisedSystem,
    labels: Labels = "values",
    index_base: int = 0,
    legend: bool = True,
) -> str:
    """Return a `tikzpicture` environment (as a string) drawing the system. Units are cm."""
    if labels not in ("values", "none"):
        raise ValueError("labels must be 'values' or 'none'")

    tower = rs.kind == "tower"
    n = rs.ndof
    values = labels == "values"
    base, kf, cf, mf, extra, app = _decode(rs)
    masses = np.diag(rs.system.M)

    # Geometry. u runs along the stack of floors, v is the orthogonal direction.
    # Tower: u = up, v = right. Chain: u = right, v = up.
    D = 2.6 if tower else 3.4  # floor spacing
    LEN_U = 0.7 if tower else 1.2  # block size along u (before mass scaling)
    SIZE_V = 2.6 if tower else 1.3  # block size along v
    V_SPRING, V_DAMP = (-0.8, 0.8) if tower else (0.35, -0.35)  # along-u elements
    T_SPRING, T_DAMP = (0.17, -0.17) if tower else (-0.2, 0.2)  # offsets of ground elements
    GD = 1.0 if tower else -1.0  # side that ground springs run towards
    REACH = 1.9  # length of a ground spring
    dirs_along = ("left", "right") if tower else ("above", "below")
    dirs_trans = ("above", "below") if tower else ("left", "right")

    def P(u: float, v: float) -> tuple[float, float]:
        return (v, u) if tower else (u, v)

    U = [(i + 1) * D for i in range(n)]
    Lb = [LEN_U * min(max(math.sqrt(mf.get(i, 1.0)), 0.8), 1.4) for i in range(n)]
    lo = lambda i: U[i] - Lb[i] / 2  # noqa: E731
    hi = lambda i: U[i] + Lb[i] / 2  # noqa: E731
    u_end = (n + 1) * D  # far wall (chain)

    out: list[str] = []

    def fmt_pt(p: tuple[float, float]) -> str:
        return f"({p[0]:.3f},{p[1]:.3f})"

    def spring(a, b, colour):
        out.append(rf"\draw[spring,{colour}] {fmt_pt(a)} -- {fmt_pt(b)};")

    def dashpot(a, b, colour):
        L = math.dist(a, b)
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        c0, c1 = 0.38 * L, 0.82 * L
        p = c0 + 0.25 * (c1 - c0)
        cw, pw = 0.14, 0.10
        out.append(rf"\begin{{scope}}[shift={{{fmt_pt(a)}}},rotate={ang:.2f},{colour}]")
        out.append(
            rf"\draw (0,0) -- ({p:.3f},0) ({p:.3f},{-pw}) -- ({p:.3f},{pw}) "
            rf"({c0:.3f},{cw}) -- ({c1:.3f},{cw}) -- ({c1:.3f},{-cw}) -- ({c0:.3f},{-cw}) "
            rf"({c1:.3f},0) -- ({L:.3f},0);"
        )
        out.append(r"\end{scope}")

    def label(a, b, text, direction, colour):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        # nudge away from the coil / dashpot body (which is ~1.4 mm either side of its axis)
        nudge = {"left": "xshift=-3pt", "right": "xshift=3pt", "above": "yshift=3pt", "below": "yshift=-3pt"}
        out.append(
            rf"\node[{direction},{nudge[direction]},font=\scriptsize,inner sep=2pt,text={colour}] "
            rf"at ({mx:.3f},{my:.3f}) {{${text}$}};"
        )

    def pair(s_pts, d_pts, k, c, col_k, col_c, dirs):
        has_k, has_c = k != 0, c != 0
        if has_k and has_c:
            spring(*s_pts, col_k)
            dashpot(*d_pts, col_c)
            if values:
                label(*s_pts, rf"k{{=}}{_fmt(k)}", dirs[0], col_k)
                label(*d_pts, rf"c{{=}}{_fmt(c)}", dirs[1], col_c)
            return
        # Only a spring, or only a damper: draw just that symbol, centred between the two lanes.
        mid = tuple(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(s_pts, d_pts))
        if has_k:
            spring(*mid, col_k)
            if values:
                label(*mid, rf"k{{=}}{_fmt(k)}", dirs[0], col_k)
        elif has_c:
            dashpot(*mid, col_c)
            if values:
                label(*mid, rf"c{{=}}{_fmt(c)}", dirs[0], col_c)

    def hatch(a, b, away, spacing=0.22, tick=0.2):
        L = math.dist(a, b)
        t = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        segs = []
        for s in range(int(L / spacing) + 1):
            q = (a[0] + t[0] * s * spacing, a[1] + t[1] * s * spacing)
            r = (q[0] + tick * (away[0] - t[0]) * 0.7, q[1] + tick * (away[1] - t[1]) * 0.7)
            segs.append(f"{fmt_pt(q)} -- {fmt_pt(r)}")
        out.append(rf"\draw[line width=1.1pt] {fmt_pt(a)} -- {fmt_pt(b)};")
        out.append(rf"\draw[line width=0.4pt] {' '.join(segs)};")

    # ---- baseline connections (spring + damper along the stack) ----------------------
    for idx, e in enumerate(base):
        if e.j is None:
            if idx == 0:
                u0, u1 = 0.0, lo(e.i)
            else:
                u0, u1 = hi(e.i), u_end
        else:
            u0, u1 = hi(min(e.i, e.j)), lo(max(e.i, e.j))
        pair(
            (P(u0, V_SPRING), P(u1, V_SPRING)),
            (P(u0, V_DAMP), P(u1, V_DAMP)),
            e.k,
            e.c,
            _status_colour(kf.get(idx, 1.0)),
            _status_colour(cf.get(idx, 1.0)),
            dirs_along,
        )

    # ---- extra ground springs (sideways for towers, downwards for chains) ------------
    v_face = GD * SIZE_V / 2
    v_wall = GD * (SIZE_V / 2 + REACH)
    for dof in sorted(extra):
        k, c = extra[dof]
        u_s, u_d = U[dof] + T_SPRING, U[dof] + T_DAMP
        pair(
            (P(u_s, v_face), P(u_s, v_wall)),
            (P(u_d, v_face), P(u_d, v_wall)),
            k,
            c,
            "msdadded",
            "msdadded",
            dirs_trans,
        )

    # ---- appendages: extra mass-spring-damper hanging off a floor ---------------------
    for attach, (ai, k, c) in sorted(app.items()):
        s = min(max(math.sqrt(masses[ai] / masses[attach]), 0.7), 1.3)  # block size vs host
        if tower and attach == n - 1:
            # on top of the tower, spring and damper side by side like the storeys
            hb, wb, reach = 0.6 * s, 1.8, 1.7
            u0, u1 = hi(attach), hi(attach) + reach
            pair(
                (P(u0, -0.5), P(u1, -0.5)),
                (P(u0, 0.5), P(u1, 0.5)),
                k, c, "msdadded", "msdadded", dirs_along,
            )
            centre, (w, h) = P(u1 + hb / 2, 0.0), (wb, hb)
        else:
            # sideways: to the left of a tower floor, above a chain mass
            side = -1.0 if tower else 1.0
            hu = (0.6 if tower else 1.3) * s  # block size along u
            wv = 1.4 if tower else 0.8  # block size along v
            v_face, v_far = side * SIZE_V / 2, side * (SIZE_V / 2 + REACH)
            u_s, u_d = U[attach] + T_SPRING, U[attach] + T_DAMP
            pair(
                (P(u_s, v_face), P(u_s, v_far)),
                (P(u_d, v_face), P(u_d, v_far)),
                k, c, "msdadded", "msdadded", dirs_trans,
            )
            centre = P(U[attach], v_far + side * wv / 2)
            w, h = (wv, hu) if tower else (hu, wv)
        txt = f"m_{{{ai + index_base}}}" + (rf"{{=}}{_fmt(masses[ai])}" if values else "")
        out.append(
            rf"\node[mass,draw=msdadded,fill=msdmadded,minimum width={w:.3f}cm,minimum height={h:.3f}cm] "
            rf"at {fmt_pt(centre)} {{${txt}$}};"
        )

    # ---- ground / walls --------------------------------------------------------------
    if tower:
        x_left = -(SIZE_V / 2 + 0.3)
        if extra:
            y_top = max(U[d] for d in extra) + 0.7
            hatch((x_left, 0.0), (v_wall, 0.0), (0, -1))
            hatch((v_wall, 0.0), (v_wall, y_top), (1, 0))
        else:
            hatch((x_left, 0.0), (SIZE_V / 2 + 0.3, 0.0), (0, -1))
        legend_xy = (x_left, -0.85)
    else:
        v_top = 0.8
        v_bot = v_wall if extra else -0.8
        hatch((0.0, v_top), (0.0, v_bot), (-1, 0))
        hatch((u_end, v_top), (u_end, v_bot), (1, 0))
        if extra:
            hatch((0.0, v_wall), (u_end, v_wall), (0, -1))
        legend_xy = (0.0, v_bot - 0.8)

    # ---- masses (drawn last so they sit on top of attachment points) ------------------
    for i in range(n):
        f = mf.get(i, 1.0)
        fill = "msdmbase" if not _changed(f) else ("msdmheavy" if f > 1 else "msdmlight")
        x, y = P(U[i], 0.0)
        w, h = (SIZE_V, Lb[i]) if tower else (Lb[i], SIZE_V)
        txt = f"m_{{{i + index_base}}}" + (rf"{{=}}{_fmt(masses[i])}" if values else "")
        out.append(
            rf"\node[mass,fill={fill},minimum width={w:.3f}cm,minimum height={h:.3f}cm] "
            rf"at ({x:.3f},{y:.3f}) {{${txt}$}};"
        )

    # ---- legend (only the categories present) -----------------------------------------
    if legend:
        items: list[tuple[str, str]] = []
        if any(_changed(v) for v in (*kf.values(), *cf.values()) if v > 1):
            items.append(("msdup", "increased k, c"))
        if any(_changed(v) for v in (*kf.values(), *cf.values()) if v < 1):
            items.append(("msddown", "decreased k, c"))
        if extra:
            items.append(("msdadded", "added ground link"))
        if app:
            items.append(("msdmadded", "added mass-spring-damper"))
        if any(v > 1 and _changed(v) for v in mf.values()):
            items.append(("msdmheavy", "heavier mass"))
        if any(v < 1 and _changed(v) for v in mf.values()):
            items.append(("msdmlight", "lighter mass"))
        x, y = legend_xy
        for colour, text in items:
            if colour.startswith("msdm"):
                edge = "msdadded" if colour == "msdmadded" else "black"  # appendages have a green outline
                out.append(
                    rf"\draw[draw={edge},fill={colour}] ({x:.3f},{y - 0.1:.3f}) "
                    rf"rectangle ({x + 0.5:.3f},{y + 0.1:.3f});"
                )
            else:
                out.append(rf"\draw[line width=1.4pt,{colour}] ({x:.3f},{y:.3f}) -- ({x + 0.5:.3f},{y:.3f});")
            out.append(
                rf"\node[anchor=west,font=\scriptsize,inner sep=1pt] at ({x + 0.6:.3f},{y:.3f}) {{{text}}};"
            )
            if tower:
                y -= 0.5  # stack vertically: a one-row legend would be wider than the tower
            else:
                x += 0.6 + 0.135 * len(text) + 0.4

    mass_font = r"\small" if tower else r"\footnotesize"
    header = [
        r"\begin{tikzpicture}[line width=0.7pt, line cap=round, line join=round,",
        r"  spring/.style={decorate,decoration={coil,aspect=0.7,segment length=2.2mm,"
        r"amplitude=1.4mm,pre length=3mm,post length=3mm}},",
        rf"  mass/.style={{draw,thick,rounded corners=1.5pt,inner sep=2pt,font={mass_font}}}]",
    ]
    header += [rf"\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}" for name, (r, g, b) in _COLOURS.items()]
    return "\n".join(header + out + [r"\end{tikzpicture}"])


def to_standalone_tex(rs: RandomisedSystem, border_pt: float = 6, **kwargs) -> str:
    """A complete .tex file (standalone class) containing just the diagram."""
    return "\n".join(
        [
            rf"\documentclass[tikz,border={border_pt:g}pt]{{standalone}}",
            TIKZ_LIBS,
            r"\begin{document}",
            to_tikz(rs, **kwargs),
            r"\end{document}",
            "",
        ]
    )


def write_tex(rs: RandomisedSystem, path: str | Path, **kwargs) -> Path:
    """Write the standalone .tex file for the diagram."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_standalone_tex(rs, **kwargs))
    return path


# --------------------------------------------------------------------------------------
# SVG export
# --------------------------------------------------------------------------------------
_CONVERTERS = {
    "pdf2svg": lambda pdf, svg: ["pdf2svg", pdf, svg],
    "pdftocairo": lambda pdf, svg: ["pdftocairo", "-svg", pdf, svg],
    "dvisvgm": lambda pdf, svg: ["dvisvgm", "--pdf", "--no-fonts", pdf, "-o", svg],
}


def _find_converter() -> str:
    for name in _CONVERTERS:
        if shutil.which(name):
            return name
    raise RuntimeError(
        "No PDF-to-SVG converter found. Install one of: pdf2svg, poppler-utils (pdftocairo) "
        "or dvisvgm (part of TeX Live / MiKTeX)."
    )


def render_svg(rs: RandomisedSystem, path: str | Path, *, keep_tex: bool = False, **kwargs) -> Path:
    """
    Compile the diagram with pdflatex and save it as an SVG at `path`.

    Args:
        keep_tex: also save the standalone .tex next to the SVG.
        **kwargs: passed to to_tikz (labels, index_base, legend) and to_standalone_tex (border_pt).
    """
    path = Path(path)
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex not found on PATH. Install a TeX distribution (TeX Live / MiKTeX).")
    converter = _find_converter()
    tex_src = to_standalone_tex(rs, **kwargs)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "msd.tex").write_text(tex_src)
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "msd.tex"],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        if run.returncode != 0 or not (tmp / "msd.pdf").exists():
            raise RuntimeError("pdflatex failed:\n" + run.stdout[-1500:])
        conv = subprocess.run(
            _CONVERTERS[converter]("msd.pdf", "msd.svg"),
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        if conv.returncode != 0 or not (tmp / "msd.svg").exists():
            raise RuntimeError(f"{converter} failed:\n{conv.stderr[-1500:]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp / "msd.svg", path)

    if keep_tex:
        path.with_suffix(".tex").write_text(tex_src)
    return path


def render_batch(
    systems: Iterable[RandomisedSystem], out_dir: str | Path, stem: str = "system", **kwargs
) -> list[Path]:
    """Render each system to `out_dir/{stem}_{i:03d}.svg` and return the paths."""
    out_dir = Path(out_dir)
    return [render_svg(rs, out_dir / f"{stem}_{i:03d}.svg", **kwargs) for i, rs in enumerate(systems)]
