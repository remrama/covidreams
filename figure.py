"""
Compile plots into a figure
"""

from pathlib import Path

import svgutils.compose as sc

import utils

# Declare filepaths for importing/exporting
derivatives_dir = Path(utils.config["derivatives_directory"])
export_path = derivatives_dir / "figure.svg"
regr_path = derivatives_dir / "regression-plot.svg"
chi2_path = derivatives_dir / "chisquared-plot.svg"
corr_path = derivatives_dir / "correlation-plot.svg"
sample_path = derivatives_dir / "samplesize-plot.svg"
methods_path = derivatives_dir / "methods.svg"

PPI = 72  # SVG units (pt) per inch
# Sample sizd plot is (3.8, 1.5)
# Regression plot is (2.8, 2)
# Chi-squared plot is (1.5, 2)
# Correlation plot is (2, 2)

fig_width = f"{7.5 * PPI}"
fig_height = f"{3.6 * PPI}"

sc.Figure(
    fig_width, fig_height,
    sc.Panel(
        sc.SVG(str(methods_path)).move(0, 10),
        sc.Text("A.", -12, 12, size=10, weight="bold", font="Arial"),
    ).move(12, 0),
    sc.Panel(
        sc.SVG(str(sample_path)),
        sc.Text("B.", -12, 12, size=10, weight="bold", font="Arial"),
    ).move(2.76 * PPI + 12 * 2 + 10, 0),
    sc.Panel(
        sc.SVG(str(regr_path)),
        sc.Text("C.", -12, 12, size=10, weight="bold", font="Arial"),
    ).move(12, 1.6 * PPI),
    sc.Panel(
        sc.SVG(str(chi2_path)),
        sc.Text("D.", -12, 12, size=10, weight="bold", font="Arial"),
    ).move(2.8 * PPI + 12 * 2 + 10, 1.6 * PPI),
    sc.Panel(
        sc.SVG(str(corr_path)),
        sc.Text("E.", -12, 12, size=10, weight="bold", font="Arial"),
    ).move(4.3 * PPI + 12 * 3 + 10 * 2, 1.6 * PPI),
).save(export_path)
