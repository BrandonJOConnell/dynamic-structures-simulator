import os
import shutil

import pytest


def latex_toolchain_available() -> bool:
    return shutil.which("pdflatex") is not None and any(
        shutil.which(c) for c in ("pdf2svg", "pdftocairo", "dvisvgm")
    )


@pytest.fixture
def latex_toolchain():
    """Skip (or, with DYNSIM_REQUIRE_LATEX=1, fail) when pdflatex or an SVG converter is missing."""
    if not latex_toolchain_available():
        msg = "pdflatex and one of pdf2svg / pdftocairo / dvisvgm are required"
        if os.environ.get("DYNSIM_REQUIRE_LATEX"):
            pytest.fail(msg)
        pytest.skip(msg)
