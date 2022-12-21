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
"""
import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns

import utils


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", default=2020, choices=[2019, 2020], type=int)
parser.add_argument("-p", "--posts", default="dreams", choices=["dreams", "wake"], type=str)
args = parser.parse_args()

year = args.year
posts = args.posts

# Declare filepaths for importing/exporting.
derivatives_dir = Path(utils.config["derivatives_directory"])
import_path_dreams = derivatives_dir / "LIWC-22 Results - r-dreams_posts - LIWC Analysis.csv"
import_path_news = derivatives_dir / "LIWC-22 Results - r-news_titles - LIWC Analysis.csv"
export_path_vals = derivatives_dir / f"{year}_{posts}_anxiety_corr-vals.tsv"
export_path_desc = derivatives_dir / f"{year}_{posts}_anxiety_corr-desc.tsv"
export_path_stat = derivatives_dir / f"{year}_{posts}_anxiety_corr-stat.tsv"
export_path_plot = derivatives_dir / f"{year}_{posts}_anxiety_corr-plot.png"

# Load data.
data = pd.read_csv(import_path_dreams)
news = pd.read_csv(import_path_news)
drms = utils.filter_flair(data, posts=posts)
df_dreams = utils.preprocess_subreddit(drms)
df_news = utils.preprocess_subreddit(news, column="title")

# Binarize COVID-19-related news.
df_news["covid"] = df_news["covid"].gt(0).astype(int)

# Merge dataframes.
df = pd.concat([df_dreams, df_news], ignore_index=True)

# Reduce to desired window.
# Have to start post-covid announcement bc otherwise there are crazy outlier jumps
# where the amount of COVID news covid news goes from like .0001 to a lot.
start_date = f"{year}-03-12"
end_date = f"{year}-09-01"
df = df.loc[df["timestamp"].between(start_date, end_date), :]

# Get weekly averages.
# (Use weekly averages bc otherwise nightmare frequency has many zeros and pct change breaks.)
weekly = (df
    .groupby(["subreddit", pd.Grouper(key="timestamp", freq="W")])
    [["covid", "emo_anx"]]
    .mean()
    .sort_index(ascending=True)
    .unstack(level=0)
    .dropna(axis=1)
    .dropna(axis=0)
    .droplevel(axis=1, level=0)
)

# Shift dreams forward to account for dream lag.
weekly["nextDreams"] = weekly["Dreams"].shift(1)

# Get percent change because time-series.
pct = weekly.pct_change()

# Combine into one dataframe.
weekly = weekly.join(pct, rsuffix="_pctchange")

# Add column indicating number of weeks post-COVID-declaration.
weekly["week"] = range(len(weekly))

# Run correlation. (rows with NaNs are automatically removed)
stat = pg.corr(weekly["news_pctchange"], weekly["nextDreams_pctchange"], method="spearman")

# Add number of samples for each, for reporting.
n_dreams, n_news = df.groupby("subreddit").size().loc[["Dreams", "news"]]
stat["n_rdreams"] = n_dreams
stat["n_news"] = n_news

# Export stats.
stat.to_csv(export_path_stat, index_label="method", sep="\t")
weekly.to_csv(export_path_vals, index_label="week", sep="\t", date_format="%Y-%m-%d")


############################################
################  Plotting  ################
############################################

# Set global matplotlib settings.
utils.load_matplotlib_settings()

# Select colormap for scatterplot.
colormap = cc.cm.CET_CBTL3_r
colornorm = plt.Normalize(vmin=1, vmax=weekly.dropna()["week"].max())

# Open figure.
fig, ax = plt.subplots(figsize=(2, 2))

# Draw data.
ax = sns.scatterplot(
    data=weekly,
    x="news_pctchange",
    y="nextDreams_pctchange",
    hue="week",
    palette=colormap,
    hue_norm=colornorm,
    legend=False,
    zorder=10,
    ax=ax,
)

# Draw correlation line.
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

# Draw stats results.
rval, pval = stat.loc["spearman", ["r", "p-val"]]
asterisks = "*" * sum( pval < cutoff for cutoff in [0.05, 0.01, 0.001] )
stats_txt = asterisks + fr"$r$ = {rval:.2f}".replace("0.", ".")
ax.text(0.07, 0.93, stats_txt, ha="left", va="top", transform=ax.transAxes)

# Adjust aesthetics.
ax.set_xlabel(r"COVID-19 news frequency ${\Delta}_{\%}$")
ax.set_ylabel(r"Next-week anxious dreaming ${\Delta}_{\%}$")
xlim = 0.35
ylim = 0.6
if posts == "wake":
    ylim += 0.2
assert not weekly["news_pctchange"].abs().ge(xlim).any()
assert not weekly["nextDreams_pctchange"].abs().ge(ylim).any()
ax.set_xlim(-xlim, xlim)
ax.set_ylim(-ylim, ylim)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

# Draw colorbar.
cax = fig.add_axes([0.67, 0.25, 0.2, 0.03])
sm = plt.cm.ScalarMappable(cmap=colormap, norm=colornorm)
cbar_ticks = [colornorm.vmin, colornorm.vmax]
cbar_ticklabels = [ str(int(x)) for x in cbar_ticks ]
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticklocation="top", ticks=[])
cbar.outline.set_linewidth(.5)
cax.text(-0.05, .5, cbar_ticklabels[0], ha="right", va="center", transform=cax.transAxes)
cax.text(1.1, .5, cbar_ticklabels[1], ha="left", va="center", transform=cax.transAxes)
cbar_label = "Weeks after\ndeclaration"
if year == 2019:
    cbar_label = cbar_label.replace("declaration", "March 11, 2019")
cbar.set_label(cbar_label)

# Export plots.
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()