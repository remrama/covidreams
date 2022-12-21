"""
Compare the frequency of nightmares posted on r/Dreams
before and during COVID.

Inputs 1 file:
    - r/Dreams LIWC output, LIWC22 dictionary run on titles

Exports 4 files:
    - chi2 data descriptives as a tsv file
    - chi2 stats as a tsv file
    - chi2 plot as a png file
    - chi2 plot as a pdf file
"""
import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", default=2020, choices=[2019, 2020], type=int)
parser.add_argument("-p", "--posts", default="dreams", choices=["dreams", "wake"], type=str)
args = parser.parse_args()

year = args.year
posts = args.posts

# Declare filepaths for importing and exporting.
derivatives_dir = Path(utils.config["derivatives_directory"])
import_path = derivatives_dir / "LIWC-22 Results - r-dreams_titles - LIWC Analysis.csv"
export_path_desc = derivatives_dir / f"{year}_{posts}_nightmares_chi2-desc.tsv"
export_path_stat = derivatives_dir / f"{year}_{posts}_nightmares_chi2-stat.tsv"
export_path_plot = derivatives_dir / f"{year}_{posts}_nightmares_chi2-plot.png"

# Load data.
data = pd.read_csv(import_path)
drms = utils.filter_flair(data, posts=posts)
df = utils.preprocess_subreddit(drms, column="title")

# Creates pandas datetimes for start, end, COVID declaration.
covid_dt = pd.to_datetime(f"{year}-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta("30D")

# Binarize nightmares.
df["nightmare"] = df["nightmare"].gt(0).astype(int)

# Reduce to the relevant time period and label pre/post-COVID.
df = df.loc[df["timestamp"].between(start_dt, end_dt, inclusive="both"), :]
df["PostCovid"] = df["timestamp"].between(covid_dt, end_dt, inclusive="both")

# Run stats.
exp, obs, stat = pg.chi2_independence(
    data=df,
    x="PostCovid",
    y="nightmare",
    correction=False,
)

# Add observed percentage of nightmares and associated confidence intervals.
obs["total"] = obs.sum(axis=1)
obs["nm_pct"] = obs[1].div(obs["total"]).mul(100)
pre_nm_vals = df.query("PostCovid==False")["nightmare"].to_numpy()
post_nm_vals = df.query("PostCovid==True")["nightmare"].to_numpy()
bootci_kwargs = dict(func="mean", method="per", n_boot=10000, decimals=6)
pre_nm_ci = 100 * pg.compute_bootci(pre_nm_vals, **bootci_kwargs)
post_nm_ci = 100 * pg.compute_bootci(post_nm_vals, **bootci_kwargs)
obs = obs.sort_index()
obs["nm_ci_lo"] = [pre_nm_ci[0], post_nm_ci[0]]
obs["nm_ci_hi"] = [pre_nm_ci[1], post_nm_ci[1]]

# Combine expected and observed frequencies into one dataframe.
desc = exp.join(obs, lsuffix="_exp", rsuffix="_obs")

# Add n to stats dataframe for easy access.
stat["n"] = desc["total"].sum()

# Export stats.
desc.to_csv(export_path_desc, sep="\t")
stat.to_csv(export_path_stat, sep="\t", index=False)


############################################
################  Plotting  ################
############################################

# Set global matplotlib settings.
utils.load_matplotlib_settings()

# Select colors.
colormap = cc.cm.cwr
pre_color = colormap(1.)
post_color = colormap(0.)
colors = [pre_color, post_color]

xvals = [0, 1]
yvals = desc["nm_pct"].to_numpy()
ci = desc[["nm_ci_lo", "nm_ci_hi"]].to_numpy()
evals = abs(yvals - ci.T)

# Open figure.
fig, ax = plt.subplots(figsize=(1.5, 2))

# Draw data.
bar_kwargs = dict(width=0.6, linewidth=1, edgecolor="black", error_kw={"lw": 1})
bars = ax.bar(xvals, yvals, yerr=evals, color=colors, **bar_kwargs)

# Draw stats results.
chi2val, pval = stat.set_index("test").loc["pearson", ["chi2", "pval"]]
asterisks = "*" * sum( pval < cutoff for cutoff in [0.05, 0.01, 0.001] )
stats_txt = asterisks + fr"$\chi^2$ = {chi2val:.1f}"
ax.text(0.5, 0.89, stats_txt, ha="left", va="bottom", transform=ax.transAxes)
hline_kwargs = dict(lw=1, color="k", capstyle="round")
ax.hlines(y=0.88, xmin=xvals[0], xmax=xvals[1], transform=ax.get_xaxis_transform(), **hline_kwargs)

# Adjust aesthetics.
xtick_labels = ["Before declaration", "After declaration"]
if year == 2019:
    xtick_labels = [ x.replace("declaration", "March 11") for x in xtick_labels ]
_, _, elines = bars.errorbar
plt.setp(elines, capstyle="round")
# ax.margins(x=0.2)
ax.set_ylim(0, 11)
ax.set_xticks(xvals)
# ax.set_xticklabels(xtick_labels)
ax.set_ylabel("Nightmare frequency (%)")
ax.tick_params(which="both", top=False, right=False, bottom=False, labelbottom=False)
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
bar_hw = bar_kwargs["width"] / 2
ax.text(0 - bar_hw, 0.2, xtick_labels[0], rotation=90, ha="right", va="bottom")
ax.text(1 - bar_hw, 0.2, xtick_labels[1], rotation=90, ha="right", va="bottom")
ax.set_xlim(-0.8, 1.5)

# Export plots.
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()