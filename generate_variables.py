"""
Generate LaTeX newcommand definitions for all in-text statistical values.

Imports 5 files:
    - regression result pkl for main analysis
    - chisquared stats tsv for main analysis
    - correlation stats tsv for main analysis
    - regression autocorrelation stats tsv
    - correlation autocorrelation stats tsv

Exports 1 file:
    - LaTeX newcommand definitions as a tex file
"""

import pandas as pd
import statsmodels.regression.linear_model as sm_lm

import utils

derivatives_dir = utils.config["derivatives_directory"]
manuscript_dir = utils.config["manuscript_directory"]
export_path = manuscript_dir / "variables.tex"


def fmt_n(x):
    """Integer with LaTeX thousands separator."""
    return f"{int(x):,}".replace(",", "{,}")


def fmt_stat(x, decimals=2):
    return f"{x:.{decimals}f}"


def fmt_p(x):
    s = f"{x:.3f}".lstrip("0")
    return r"<{.001}" if s == ".000" else s


def fmt_r(x):
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else s  # 0.45 -> .45


def newcommand(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


# --- Load data ---

regr = sm_lm.RegressionResultsWrapper.load(derivatives_dir / "regression-results.pkl")
chi2_stat = pd.read_table(derivatives_dir / "chisquared-stat.tsv", index_col="test").loc["pearson"]
corr_stat = pd.read_table(derivatives_dir / "correlation-stat.tsv", index_col="test").loc["spearman"]
regr_acor = pd.read_table(derivatives_dir / "regression-acor.tsv", index_col="test")
corr_acor = pd.read_table(derivatives_dir / "correlation-acor.tsv", index_col="test")

# --- Build commands ---

# Sample sizes
main_n = newcommand("mainN", fmt_n(chi2_stat["n"]))

# Regression (ITS) result for Covid coefficient
b2 = regr.params["Covid"]
b2_p = regr.pvalues["Covid"]
b2_ci = regr.conf_int().loc["Covid"]
b2_lower, b2_upper = b2_ci
regr_result = newcommand(
    "regrResult",
    rf"$B = {fmt_stat(b2)}$, $CI = [{fmt_stat(b2_lower)}, {fmt_stat(b2_upper)}]$, $p = {fmt_p(b2_p)}$",
)

# Chi-squared result
chi_val = chi2_stat["chi2"]
chi_dof = int(chi2_stat["dof"])
chi_n = int(chi2_stat["n"])
chi_p = chi2_stat["pval"]
chi_result = newcommand(
    "chiResult",
    rf"$\chi^2({chi_dof}, N = {fmt_n(chi_n)}) = {fmt_stat(chi_val)}$, $p = {fmt_p(chi_p)}$",
)

# Correlation result
r_val = corr_stat["r"]
r_lo = corr_stat["r_lower"]
r_hi = corr_stat["r_upper"]
r_p = corr_stat["pval"]
r_df = int(corr_stat["n"]) - 2
corr_result = newcommand(
    "corrResult",
    rf"$r_{{{r_df}}} = {fmt_r(r_val)}$, $CI = [{fmt_r(r_lo)}, {fmt_r(r_hi)}]$, $p = {fmt_p(r_p)}$",
)

# Correlation sample sizes
corr_n_dreams = newcommand("corrNDreams", fmt_n(corr_stat["n_rdreams"]))
corr_n_news   = newcommand("corrNNews",   fmt_n(corr_stat["n_rnews"]))
corr_n_weeks  = newcommand("corrNWeeks",  str(int(corr_stat["n"])))

# Regression autocorrelation
regr_dw = newcommand(
    "regrDurbinWatson",
    rf"$d = {fmt_stat(regr_acor.at['Durbin-Watson', 'stat'])}$",
)
regr_bg_nlags = int(regr_acor.at["Breusch-Godfrey", "nlags"])
regr_bg = newcommand(
    "regrBreuschGodfrey",
    rf"$\chi^2({regr_bg_nlags}) = {fmt_stat(regr_acor.at['Breusch-Godfrey', 'stat'])}$, $p = {fmt_p(regr_acor.at['Breusch-Godfrey', 'pval'])}$",
)
regr_lb_nlags = int(regr_acor.at["Ljung-Box", "nlags"])
regr_lb = newcommand(
    "regrLjungBox",
    rf"$Q({regr_lb_nlags}) = {fmt_stat(regr_acor.at['Ljung-Box', 'stat'])}$, $p = {fmt_p(regr_acor.at['Ljung-Box', 'pval'])}$",
)

# Correlation autocorrelation
corr_dw = newcommand(
    "corrDurbinWatson",
    rf"$d = {fmt_stat(corr_acor.at['Durbin-Watson', 'stat'])}$",
)
corr_bg_nlags = int(corr_acor.at["Breusch-Godfrey", "nlags"])
corr_bg = newcommand(
    "corrBreuschGodfrey",
    rf"$\chi^2({corr_bg_nlags}) = {fmt_stat(corr_acor.at['Breusch-Godfrey', 'stat'])}$, $p = {fmt_p(corr_acor.at['Breusch-Godfrey', 'pval'])}$",
)
corr_lb_nlags = int(corr_acor.at["Ljung-Box", "nlags"])
corr_lb = newcommand(
    "corrLjungBox",
    rf"$Q({corr_lb_nlags}) = {fmt_stat(corr_acor.at['Ljung-Box', 'stat'])}$, $p = {fmt_p(corr_acor.at['Ljung-Box', 'pval'])}$",
)

# --- Export ---

commands = [
    "% Sample sizes",
    main_n,
    "",
    "% Regression (ITS) result",
    regr_result,
    "",
    "% Chi-squared result",
    chi_result,
    "",
    "% Correlation result",
    corr_result,
    corr_n_dreams,
    corr_n_news,
    corr_n_weeks,
    "",
    "% Regression autocorrelation diagnostics",
    regr_dw,
    regr_bg,
    regr_lb,
    "",
    "% Correlation autocorrelation diagnostics",
    corr_dw,
    corr_bg,
    corr_lb,
]

with open(export_path, "w", encoding="utf-8") as f:
    f.write("\n".join(commands) + "\n")
