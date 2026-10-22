"""
Simulate a randomised system and check its spectrum against the true natural frequencies.

The randomised system's modal properties are known exactly from its matrices, so this is the
kind of ground-truth check an operational modal analysis (OMA) method can be tested against.

Needs matplotlib:  uv sync --extra examples   (or: pip install "dynsim[examples]")

    uv run python examples/simulate_and_check_modes.py            # saves docs/img/spectrum_example.png
    uv run python examples/simulate_and_check_modes.py --show     # also opens a window
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from dynsim import (
    AppendageSpec,
    GroundDamperSpec,
    GroundSpringSpec,
    ScaleConnectionSpec,
    ScaleMassSpec,
    make_state_matrices,
    sample_system,
    simulate_discrete_ssm,
    van_loan_discretisation,
)

# Colours follow the reference data-viz palette (series 1 blue, neutral ink and grid)
SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="open the figure in a window as well")
default_out = Path(__file__).resolve().parents[1] / "docs" / "img" / "spectrum_example.png"
parser.add_argument("--out", type=Path, default=default_out)
args = parser.parse_args()

# 1. A randomised system: the same draw as the README gallery's chain
rs = sample_system(
    111,
    "chain",
    5,
    feature_types=[
        ScaleConnectionSpec(n=2, k_range=(0.5, 2.0), c_range=(0.5, 2.0)),
        GroundSpringSpec(n=1),
        GroundDamperSpec(n=1),
        ScaleMassSpec(n=(1, 2)),
        AppendageSpec(attach="random"),
    ],
)
system = rs.system
n = system.ndof  # 6: five floors plus one appendage
print("\n".join(rs.describe()))

# 2. Exact modal properties from the matrices
modal = system.modal
print("true natural frequencies (Hz):", np.round(modal.wn, 3))

# 3. Simulate accelerations under white-noise force on every dof
fs = 40.0
F, C = make_state_matrices(system, observe="a")
L = np.vstack([np.zeros((n, n)), np.linalg.inv(system.M)])  # force noise enters through M^-1
A, Q, _ = van_loan_discretisation(F, 1.0, L, 1 / fs)  # noise spectral density 1 N^2/Hz
_, y = simulate_discrete_ssm(A, C, Q, 1e-8 * np.eye(n), np.zeros(2 * n), T=2**17, rng=0)

# 4. Spectrum, averaged over channels
f, pxx = welch(y, fs=fs, nperseg=2**12)
psd = pxx.mean(axis=0)

# 5. Plot: one series, so no legend; the true frequencies are marked and labelled directly
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.semilogy(f, psd, color=SERIES, lw=1.2)
ax.set_xlim(0, modal.wn.max() * 1.25)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 6)  # headroom for the mode labels
for j, wn in enumerate(modal.wn, start=1):
    ax.axvline(wn, color=INK_2, lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.text(wn, ax.get_ylim()[1] * 0.8, f"{j}", ha="center", va="top", color=INK_2, fontsize=8,
            bbox={"fc": SURFACE, "ec": "none", "pad": 1.5})
ax.set_xlabel("Frequency (Hz)", color=INK_2)
ax.set_ylabel("Acceleration PSD, mean over channels\n((m/s²)²/Hz)", color=INK_2)
ax.set_title("Simulated spectrum vs. true natural frequencies (dashed, numbered by mode)",
             loc="left", color=INK, fontsize=10)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.tick_params(colors=INK_2, length=0, which="both")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.tight_layout()

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, facecolor=SURFACE)
print(f"saved {args.out}")
if args.show:
    plt.show()
