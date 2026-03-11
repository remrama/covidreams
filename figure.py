"""
Compile plots into a multi-panel figure
"""

from pathlib import Path

import cairosvg
import svgutils.compose as sc

import utils

# Declare filepaths for importing/exporting
derivatives_dir = utils.config["derivatives_directory"]
export_path = derivatives_dir / "figure.svg"
regr_path = derivatives_dir / "regression-plot.svg"
chi2_path = derivatives_dir / "chisquared-plot.svg"
corr_path = derivatives_dir / "correlation-plot.svg"
sample_path = derivatives_dir / "samplesize-plot.svg"
methods_path = Path("../docs") / "methods.svg"

PPI = 72  # SVG units (pt) per inch

# The figure size of all plots in inches (width, height)
METHODS_SIZE = (2.76, 1.5)
SAMPLE_SIZE = (3.8, 1.5)
REGR_SIZE = (2.8, 2)
CHI2_SIZE = (1.5, 2)
CORR_SIZE = (2, 2)

methods_x = METHODS_SIZE[0] * PPI
sample_x = SAMPLE_SIZE[0] * PPI
regr_x = REGR_SIZE[0] * PPI
chi2_x = CHI2_SIZE[0] * PPI
corr_x = CORR_SIZE[0] * PPI

TEXT_KWARGS = {"size": 10, "weight": "bold", "font": "Times New Roman"}
TEXT_ARGS = (-12, 12)  # x and y offsets for text labels
TEXT_HPAD = 12  # horizontal padding for letters
VPAD = 1.6  # vertical padding between panel rows
HPAD = 10  # horizontal padding between panels in the same row
vpad_ppi = VPAD * PPI

fig_width = f"{7.25 * PPI}"
fig_height = f"{3.6 * PPI}"

sc.Figure(
    fig_width,
    fig_height,
    # Top row
    sc.Panel(
        sc.SVG(methods_path).move(0, 10),
        sc.Text("A.", *TEXT_ARGS, **TEXT_KWARGS),
    ).move(TEXT_HPAD, 0),
    sc.Panel(
        sc.SVG(sample_path),
        sc.Text("B.", *TEXT_ARGS, **TEXT_KWARGS),
    ).move(methods_x + TEXT_HPAD * 2 + HPAD, 0),
    # Bottom row
    sc.Panel(
        sc.SVG(regr_path),
        sc.Text("C.", *TEXT_ARGS, **TEXT_KWARGS),
    ).move(TEXT_HPAD, vpad_ppi),
    sc.Panel(
        sc.SVG(chi2_path),
        sc.Text("D.", *TEXT_ARGS, **TEXT_KWARGS),
    ).move(regr_x + TEXT_HPAD * 2 + HPAD, vpad_ppi),
    sc.Panel(
        sc.SVG(corr_path),
        sc.Text("E.", *TEXT_ARGS, **TEXT_KWARGS),
    ).move(regr_x + chi2_x + TEXT_HPAD * 3 + HPAD * 2, vpad_ppi),
).save(export_path)

cairosvg.svg2pdf(url=str(export_path), write_to=str(export_path.with_suffix(".pdf")))
export_path.unlink()
