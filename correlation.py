"""
Correlate the frequency of COVID-19-related news headlines
on r/news with the frequency of LIWC anxiety words in dream
reports on r/Dreams.

Inputs 2 files:
    - r/news LIWC output, custom COVID dictionary run on post titles
    - r/Dreams LIWC output, LIWC22 dictionary run on posts

Exports 4 files:
    - values being correlated as a tsv file
    - correlation stats as a tsv file
    - correlation plot as a png file
    - correlation plot as a pdf file
    - autocorrelation check plot and stats as a png file
    - autocorrelation check plot and stats as a pdf file
"""

import argparse

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prioryear", action="store_true", help="Run on 2019 data instead of 2020 data."
)
parser.add_argument(
    "--nondreams", action="store_true", help="Run on wake text instead of dream text."
)
args = parser.parse_args()

year = 2019 if args.prioryear else 2020
flair = "nondreams" if args.nondreams else "dreams"

# Declare filepaths for importing/exporting
derivatives_dir = utils.config["derivatives_directory"]
export_parent = derivatives_dir
if year == 2019:
    export_parent = export_parent / "prioryear"
if flair == "nondreams":
    export_parent = export_parent / "nondreams"
export_parent.mkdir(exist_ok=True)
export_path_vals = export_parent / "correlation-vals.tsv"
export_path_desc = export_parent / "correlation-desc.tsv"
export_path_stat = export_parent / "correlation-stat.tsv"
export_path_plot = export_parent / "correlation-plot.png"
export_path_acor = export_parent / "correlation-acor.png"
export_path_acor_before = export_parent / "correlation-acor_before.png"

# Load data
drms = utils.read_liwc_csv(subreddit="dreams", dream_filter=flair)
news = utils.read_liwc_csv(subreddit="news")
drms["subreddit"] = "Dreams"
news["subreddit"] = "news"

# Merge dataframes
df = pd.concat([drms, news], ignore_index=True)

# Reduce to desired window
# Have to start post-covid announcement bc otherwise there are crazy outlier jumps
# where the amount of COVID news covid news goes from like .0001 to a lot.
start_date = pd.to_datetime(f"{year}-03-12", utc=True)
end_date = pd.to_datetime(f"{year}-09-01", utc=True)
df = df.loc[df["timestamp"].between(start_date, end_date), :]

# Get weekly averages
# (Use weekly averages bc otherwise nightmare frequency has many zeros and pct change breaks)
weekly = (
    df.groupby(["subreddit", pd.Grouper(key="timestamp", freq="W")])[
        ["covid", "anxiety"]
    ]
    .mean()
    .sort_index(ascending=True)
    .unstack(level=0)
    .dropna(axis=1)
    .dropna(axis=0)
    .droplevel(axis=1, level=0)
)

# Shift dreams forward to account for retrospective dream reporting
weekly["nextDreams"] = weekly["Dreams"].shift(1)

# Get percent change because time-series
pct = weekly.pct_change()

# Combine into one dataframe
weekly = weekly.join(pct, rsuffix="_pctchange")

# Add column indicating number of weeks post-COVID-declaration
weekly["weeks_after"] = range(len(weekly))

# Run correlation (rows with NaNs are automatically removed)
stat = pg.corr(
    weekly["news_pctchange"], weekly["nextDreams_pctchange"], method="spearman"
)

# Add number of samples for each, for reporting
n_dreams, n_news = df.groupby("subreddit").size().loc[["Dreams", "news"]]
stat["n_rdreams"] = n_dreams
stat["n_rnews"] = n_news

# Export stats
stat.to_csv(export_path_stat, index_label="method", sep="\t")
weekly.to_csv(
    export_path_vals, index_label="week", sep="\t", na_rep="N/A", date_format="%Y-%m-%d"
)

############################################
################  Plotting  ################
############################################

# Set global matplotlib settings
utils.load_matplotlib_settings()

# Select colormap for scatterplot
colormap = cc.cm.CET_CBTL3_r
colornorm = plt.Normalize(vmin=1, vmax=weekly.dropna()["weeks_after"].max())

# Open figure
fig, ax = plt.subplots(figsize=(2, 2))

# Draw data
ax = sns.scatterplot(
    data=weekly,
    x="news_pctchange",
    y="nextDreams_pctchange",
    hue="weeks_after",
    palette=colormap,
    hue_norm=colornorm,
    legend=False,
    zorder=10,
    ax=ax,
)

# Draw correlation line
ax = sns.regplot(
    data=weekly,
    x="news_pctchange",
    y="nextDreams_pctchange",
    color="black",
    scatter=False,
    ci=95,
    n_boot=2000,
    seed=1,
    ax=ax,
)

# Draw stats results
rval, pval = stat.loc["spearman", ["r", "p_val"]]
asterisks = "*" * sum(pval < cutoff for cutoff in [0.05, 0.01, 0.001])
stats_txt = asterisks + rf"$r$ = {rval:.2f}".replace("0.", ".")
text_x = 0.6 if year == 2019 else 0.07
ax.text(text_x, 0.93, stats_txt, ha="left", va="top", transform=ax.transAxes)

# Adjust aesthetics
ax.set_xlabel(r"COVID-19 news frequency ${\Delta}_{\%}$")
ax.set_ylabel(r"Next-week anxious dreaming ${\Delta}_{\%}$")
xlim = 0.35
ylim = 0.6
if flair == "nondreams":
    ylim += 0.2
# Bizarre situation where there is an outlier week in 2019 that coincidentally has tones of covid words in it.
if year == 2019:
    xmin, xmax = -0.75, 1.9
    assert not weekly["news_pctchange"].le(xmin).any()
    assert not weekly["news_pctchange"].ge(xmax).any()
    ax.set_xlim(xmin, xmax)
else:
    assert not weekly["news_pctchange"].abs().ge(xlim).any()
    ax.set_xlim(-xlim, xlim)
assert not weekly["nextDreams_pctchange"].abs().ge(ylim).any()
ax.set_ylim(-ylim, ylim)
if year == 2019:
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
else:
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

# Draw colorbar
cax = fig.add_axes([0.67, 0.25, 0.2, 0.03])
smap = plt.cm.ScalarMappable(cmap=colormap, norm=colornorm)
cbar_ticks = [colornorm.vmin, colornorm.vmax]
cbar_ticklabels = [str(int(x)) for x in cbar_ticks]
cbar = fig.colorbar(
    smap, cax=cax, orientation="horizontal", ticklocation="top", ticks=[]
)
cbar.outline.set_linewidth(0.5)
cax.text(
    -0.05, 0.5, cbar_ticklabels[0], ha="right", va="center", transform=cax.transAxes
)
cax.text(1.1, 0.5, cbar_ticklabels[1], ha="left", va="center", transform=cax.transAxes)
cbar_label = "Weeks after\ndeclaration"
if year == 2019:
    cbar_label = cbar_label.replace("declaration", "March 11, 2019")
cbar.set_label(cbar_label)

# Export plots
utils.save_and_close_fig(export_path_plot, include_svg=True)

#######################################################################################
################  Stats and Plotting for Autocorrelation/Stationarity  ################
#######################################################################################
# Test 4 time-series for autocorrelation and stationarity,
# both subreddits (news and Dreams) and both stages of processing (raw and percent change).

# Open up figure
fig, axes = plt.subplots(
    2, 2, figsize=(6, 6), constrained_layout=True, sharex=True, sharey=True
)
# Select universal plotting keyword arguments
acf_kwargs = dict(
    alpha=0.05,
    zero=True,
    missing="drop",
    title=None,
    bartlett_confint=False,
    clip_on=False,
)

for col, subreddit in enumerate(["news", "Dreams"]):
    for row, stage in enumerate(["raw", "pctchange"]):
        ax = axes[row, col]
        column = f"{subreddit}_{stage}" if stage == "pctchange" else subreddit
        data = weekly[column].dropna().to_numpy()
        title = f"COVID-19 on r/{subreddit}, {stage}"
        if stage == "pctchange":
            title = title.replace(stage, r"${\Delta}_{\%}$")

        # Ljung-Box Q-test for autocorrelation
        nlags_lb = 10
        lb = sm.stats.acorr_ljungbox(data, lags=nlags_lb)
        lb_stat = lb.at[nlags_lb, "lb_stat"]
        lb_p = lb.at[nlags_lb, "lb_pvalue"]
        # Compile all stats into text to write on the plots
        strings = [
            f"Ljung-Box (lag={nlags_lb}) = {lb_stat:.1f}, p = {lb_p:.3f}",
        ]
        strings = [
            s.replace("p = 0.", "p = .").replace("p = .000", "p < .001")
            for s in strings
        ]
        text = "\n".join(strings)
        text_pass = "\n".join(
            [
                "PASS" if lb_p > 0.05 else "FAIL",
            ]
        )
        # Draw an ACF plot/correlogram to visually inspect autocorrelation
        sm.graphics.tsa.plot_acf(data, ax, **acf_kwargs)
        # Draw text
        ax.text(
            0.5,
            0.95,
            title,
            ha="center",
            va="top",
            weight="bold",
            transform=ax.transAxes,
        )
        ax.text(0.83, 0.05, text, ha="right", va="bottom", transform=ax.transAxes)
        ax.text(0.85, 0.05, text_pass, ha="left", va="bottom", transform=ax.transAxes)

# Export plots
utils.save_and_close_fig(export_path_plot)

#########################################################

# Run regression
# Run correlation (rows with NaNs are automatically removed)
weekly_nonan = weekly.dropna(how="any", axis="rows")
model = sm.formula.ols(
    formula="nextDreams_pctchange ~ news_pctchange", data=weekly_nonan
)
model = model.fit()

nlags_lb = 10
nlags_bg = 2
acf_kwargs = dict(
    lags=nlags_lb,
    alpha=0.05,
    zero=True,
    missing="drop",
    title=None,
    bartlett_confint=False,
    clip_on=True,
)
# Ljung-Box Q-test
lb = sm.stats.acorr_ljungbox(model.resid, lags=nlags_lb)
lb_stat = lb.at[nlags_lb, "lb_stat"]
lb_p = lb.at[nlags_lb, "lb_pvalue"]
# Breusch-Godfrey test
lm_stat, lm_p, f_stat, f_p = sm.stats.acorr_breusch_godfrey(model, nlags=nlags_bg)
# ACF plot, visual inspection
strings = [
    f"Ljung-Box (lag={nlags_lb}) = {lb_stat:.1f}, p = {lb_p:.3f}",
    f"Breusch-Godfrey (lag={nlags_bg}) = {lm_stat:.1f}, p = {lm_p:.3f}",
]
strings = [
    s.replace("p = 0.", "p = .").replace("p = .000", "p < .001") for s in strings
]
text = "\n".join(strings)
text_pass = "\n".join(
    [
        "PASS" if lb_p > 0.05 else "FAIL",
        "PASS" if lm_p > 0.05 else "FAIL",
    ]
)
# Open up figure
fig, ax = plt.subplots(
    figsize=(3, 3), constrained_layout=True, sharex=True, sharey=True
)
# Draw an ACF plot/correlogram to visually inspect autocorrelation
sm.graphics.tsa.plot_acf(model.resid, ax, **acf_kwargs)
# Draw text
title = "Autocorrelation and stationarity\nin model residuals"
ax.text(0.5, 0.95, title, ha="center", va="top", weight="bold", transform=ax.transAxes)
ax.text(0.83, 0.05, text, ha="right", va="bottom", transform=ax.transAxes)
ax.text(0.85, 0.05, text_pass, ha="left", va="bottom", transform=ax.transAxes)

# Export plots
utils.save_and_close_fig(export_path_plot)
plt.close()
