import shutil
import xml.dom.minidom

import pytest

from dynsim import (
    AddAppendage,
    AddGroundDamper,
    AddGroundSpring,
    ScaleConnection,
    ScaleMass,
    build_system,
    render_batch,
    render_svg,
    sample_system,
    to_standalone_tex,
    to_tikz,
)

FULL = [
    ScaleConnection(1, 1.8),
    ScaleConnection(3, 0.5),
    AddGroundSpring(2, 1500, 1.2),
    ScaleMass(0, 2.0),
    ScaleMass(3, 0.6),
    AddAppendage(4, 1.5, 900, 0.7),
]


@pytest.mark.parametrize("kind", ["tower", "chain"])
def test_tikz_structure(kind):
    rs = build_system(kind, 5, features=FULL)
    tex = to_tikz(rs)
    assert tex.startswith(r"\begin{tikzpicture}") and tex.rstrip().endswith(r"\end{tikzpicture}")
    assert tex.count(r"\node[mass") == rs.system.ndof  # every mass, including the appendage
    assert tex.count(r"\begin{scope}") == tex.count(r"\end{scope}")
    for entry in ("increased k, c", "decreased k, c", "added ground link", "added mass-spring-damper",
                  "heavier mass", "lighter mass"):
        assert entry in tex


@pytest.mark.parametrize("kind", ["tower", "chain"])
def test_baseline_has_no_legend_and_labels_can_be_switched_off(kind):
    rs = build_system(kind, 4)
    tex = to_tikz(rs)
    for entry in ("increased k, c", "decreased k, c", "added ground link", "added mass-spring-damper",
                  "heavier mass", "lighter mass"):
        assert entry not in tex
    assert "k{=}" in tex and "k{=}" not in to_tikz(rs, labels="none")

    heavy = build_system(kind, 4, features=[ScaleMass(0, 2.0)])
    assert "heavier mass" in to_tikz(heavy)
    assert "heavier mass" not in to_tikz(heavy, legend=False)


def test_index_base():
    rs = build_system("tower", 3)
    assert "m_{0}" in to_tikz(rs) and "m_{3}" not in to_tikz(rs)
    assert "m_{3}" in to_tikz(rs, index_base=1) and "m_{0}" not in to_tikz(rs, index_base=1)


def test_parallel_ground_springs_are_merged():
    rs = build_system("tower", 3, features=[AddGroundSpring(1, 500, 0.5), AddGroundSpring(1, 500, 0.5)])
    assert "k{=}1000" in to_tikz(rs)


def test_unsupported_configurations_raise():
    with pytest.raises(ValueError, match="at most one appendage per floor"):
        to_tikz(build_system("tower", 3, features=[AddAppendage(2, 1, 1), AddAppendage(2, 1, 1)]))
    with pytest.raises(ValueError, match="baseline floors"):
        to_tikz(build_system("tower", 3, features=[AddAppendage(2, 1, 1), AddAppendage(3, 1, 1)]))
    with pytest.raises(ValueError, match="labels"):
        to_tikz(build_system("tower", 3), labels="some")


def test_standalone_document():
    tex = to_standalone_tex(build_system("chain", 3), border_pt=4)
    assert r"\documentclass[tikz,border=4pt]{standalone}" in tex
    assert r"\usetikzlibrary{decorations.pathmorphing}" in tex and r"\end{document}" in tex


def test_missing_toolchain_gives_clear_errors(monkeypatch, tmp_path):
    rs = build_system("tower", 3)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="pdflatex"):
        render_svg(rs, tmp_path / "a.svg")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pdflatex" if name == "pdflatex" else None)
    with pytest.raises(RuntimeError, match="converter"):
        render_svg(rs, tmp_path / "a.svg")


@pytest.mark.latex
@pytest.mark.parametrize("kind", ["tower", "chain"])
def test_render_svg_compiles(latex_toolchain, tmp_path, kind):
    rs = sample_system(11, kind, 5, feature_types=["scale", "ground", "mass", "appendage"])
    out = render_svg(rs, tmp_path / "nested" / f"{kind}.svg", keep_tex=True)
    assert out.exists() and out.with_suffix(".tex").exists()
    assert "<svg" in out.read_text()
    xml.dom.minidom.parse(str(out))


@pytest.mark.latex
def test_render_batch(latex_toolchain, tmp_path):
    systems = [sample_system(s, "tower", 4) for s in range(3)]
    paths = render_batch(systems, tmp_path, stem="t")
    assert [p.name for p in paths] == ["t_000.svg", "t_001.svg", "t_002.svg"]
    assert all(p.exists() for p in paths)


def n_labels(tex: str) -> tuple[int, int]:
    return tex.count("k{=}"), tex.count("c{=}")


def test_spring_only_and_damper_only_elements_are_drawn_without_the_missing_symbol():
    plain = n_labels(to_tikz(build_system("tower", 4)))  # one spring and one damper per connection
    assert plain == (4, 4)
    # a ground spring with no damper adds only a spring; a ground damper adds only a damper
    assert n_labels(to_tikz(build_system("tower", 4, features=[AddGroundSpring(2, 900, 0.0)]))) == (5, 4)
    assert n_labels(to_tikz(build_system("tower", 4, features=[AddGroundDamper(2, 1.0)]))) == (4, 5)
    # both on one floor are parallel elements: one spring and one damper
    both = build_system("tower", 4, features=[AddGroundSpring(2, 900, 0.0), AddGroundDamper(2, 1.0)])
    assert n_labels(to_tikz(both)) == (5, 5)
    # an appendage without a damper is drawn as a spring only
    assert n_labels(to_tikz(build_system("chain", 3, features=[AddAppendage(2, 1.0, 500.0, 0.0)]))) == (5, 4)
    assert "c{=}0" not in to_tikz(build_system("tower", 4, features=[AddGroundSpring(2, 900, 0.0)]))


def test_damping_only_change_colours_only_the_damper():
    tex = to_tikz(build_system("tower", 3, features=[ScaleConnection(1, 1.0, 2.0)]))
    assert "text=msdup" in tex and "text=msddown" not in tex
    assert tex.count("text=msdup") == 1  # one label (the damper); the spring label stays black


@pytest.mark.latex
def test_render_spring_only_and_damper_only(latex_toolchain, tmp_path):
    features = [AddGroundSpring(0, 900, 0.0), AddGroundDamper(2, 1.0), ScaleConnection(1, 1.0, 0.4)]
    rs = build_system("chain", 4, features=features)
    assert "<svg" in render_svg(rs, tmp_path / "x.svg").read_text()
