"""
Compile statistics for manuscript Table 1.

Imports 9 files:
    - regression result pkl for main, prioryear, and nondreams analyses
    - correlation stats tsv for main, prioryear, and nondreams analyses
    - chi-squared stats tsv for main, prioryear, and nondreams analyses

Exports 1 file:
    - LaTeX-formatted Table 1 as a tex file
"""

import pandas as pd
import statsmodels.regression.linear_model as sm_lm

import utils

derivatives_dir = utils.config["derivatives_directory"]
tables_dir = utils.config["tables_directory"]
export_path = tables_dir / "table1.tex"


def fmt_beta(x):
    s = f"{x:.2f}"
    return s.replace("-0.00", "0.00")


def fmt_r(x):
    s = f"{x:.2f}"
    if s.startswith("0."):
        return s[1:]  # strip leading zero: 0.45 -> .45
    return s


def fmt_chi2(x):
    return f"{x:.2f}"


def fmt_p(x):
    s = f"{x:.3f}".lstrip("0")
    return "<.001" if s == ".000" else s


def bold(s):
    return rf"\textbf{{{s}}}"


def bold_if_sig(val_str, p):
    return bold(val_str) if p < 0.05 else val_str


def load_regression(pkl_path):
    result = sm_lm.RegressionResultsWrapper.load(pkl_path)
    return {
        "b1": result.params["Time"],
        "b2": result.params["Covid"],
        "b3": result.params["TimeCovid"],
        "p1": result.pvalues["Time"],
        "p2": result.pvalues["Covid"],
        "p3": result.pvalues["TimeCovid"],
    }


def load_correlation(tsv_path):
    stat = pd.read_table(tsv_path, index_col="test")
    row = stat.loc["spearman"]
    return {"r": row["r"], "p": row["pval"]}


def load_chisquared(tsv_path):
    stat = pd.read_table(tsv_path, index_col="test")
    row = stat.loc["pearson"]
    return {"chi2": row["chi2"], "p": row["pval"]}


def build_row(label, reg, chi2, corr):
    b1 = bold_if_sig(fmt_beta(reg["b1"]), reg["p1"])
    b2 = bold_if_sig(fmt_beta(reg["b2"]), reg["p2"])
    b3 = bold_if_sig(fmt_beta(reg["b3"]), reg["p3"])
    p1 = bold_if_sig(fmt_p(reg["p1"]), reg["p1"])
    p2 = bold_if_sig(fmt_p(reg["p2"]), reg["p2"])
    p3 = bold_if_sig(fmt_p(reg["p3"]), reg["p3"])
    r  = bold_if_sig(fmt_r(corr["r"]), corr["p"])
    pr = bold_if_sig(fmt_p(corr["p"]), corr["p"])
    x  = bold_if_sig(fmt_chi2(chi2["chi2"]), chi2["p"])
    px = bold_if_sig(fmt_p(chi2["p"]), chi2["p"])
    return " & ".join([label, b1, p1, b2, p2, b3, p3, r, pr, x, px]) + r" \\"


rows = [
    build_row(
        "Dreams, 2020",
        load_regression(derivatives_dir / "regression-results.pkl"),
        load_chisquared(derivatives_dir / "chisquared-stat.tsv"),
        load_correlation(derivatives_dir / "correlation-stat.tsv"),
    ),
    build_row(
        "Dreams, 2019",
        load_regression(derivatives_dir / "prioryear" / "regression-results.pkl"),
        load_chisquared(derivatives_dir / "prioryear" / "chisquared-stat.tsv"),
        load_correlation(derivatives_dir / "prioryear" / "correlation-stat.tsv"),
    ),
    build_row(
        "Non-dreams, 2020",
        load_regression(derivatives_dir / "nondreams" / "regression-results.pkl"),
        load_chisquared(derivatives_dir / "nondreams" / "chisquared-stat.tsv"),
        load_correlation(derivatives_dir / "nondreams" / "correlation-stat.tsv"),
    ),
]

col_widths = (
    r"p{0.17\textwidth} "
    r"p{0.04\textwidth} p{0.06\textwidth} p{0.04\textwidth} p{0.06\textwidth} p{0.04\textwidth} p{0.07\textwidth} "
    r"p{0.04\textwidth} p{0.04\textwidth} "
    r"p{0.04\textwidth} p{0.07\textwidth}"
)

header1 = (
    r"    {} "
    r"& \multicolumn{6}{l}{Anxiety regression} "
    r"& \multicolumn{2}{l}{Nightmares chi-sq.} "
    r"& \multicolumn{2}{l}{Anxiety corr.} "
    r"\\"
)

header2 = (
    r"    Text source "
    r"& $B_1$ & $p$ & $B_2$ & $p$ & $B_3$ & $p$ "
    r"& $r$ & $p$ "
    r"& $\chi^2$ & $p$ "
    r"\\"
)

lines = [
    rf"\begin{{tabular}}{{{col_widths}}}",
    r"    \toprule",
    header1,
    header2,
    r"    \midrule",
    *[f"    {row}" for row in rows],
    r"    \bottomrule",
    r"\end{tabular}",
]
tex = "\n".join(lines)

with open(export_path, "w", encoding="utf-8") as f:
    f.write(tex)
