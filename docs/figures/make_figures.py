#!/usr/bin/env python3
"""Generate the vector-PDF figures for docs/experiments.tex.

All data is transcribed from the experiment reports in experiments/.
Each figure cites its source report in a comment; one fig_<name>()
function per figure. Run from anywhere: outputs land next to this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent

# Consistent style
plt.rcParams.update(
    {
        "figure.figsize": (6.0, 3.5),
        "figure.constrained_layout.use": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

ACCENT = "#2563eb"  # blue
GOOD = "#16a34a"  # green
BAD = "#dc2626"  # red
MUTED = "#9ca3af"  # gray

# Domain style header — keep axis units/labels going through these instead
# of hardcoding the unit per figure (values mirror [metrics] in domain.toml).
PRIMARY_METRIC = "MAE"
VALUE_UNIT = "units"
VALUE_AXIS = f"{PRIMARY_METRIC} ({VALUE_UNIT})"
kfmt = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {name}")


# Figures are added by the report-curator agent, one fig_<name>() per
# figure, each citing its source report.

if __name__ == "__main__":
    pass
