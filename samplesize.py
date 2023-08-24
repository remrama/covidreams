"""
Plot the amount of r/Dreams posts over time and total.

Imports 1 file:
    - raw r/Dreams data

Exports 3 files:
    - descriptives for daily sample sizes as tsv file
    - sample size plot as a png file
    - sample size plot as a pdf file
"""
import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", default="2020", choices=["2019", "2020"], type=str)
args = parser.parse_args()

year = args.year

# Declare filepaths for importing/exporting.
import_dir = Path(utils.config["source_directory"])
export_dir = Path(utils.config["derivatives_directory"])
import_path = import_dir / "r-dreams.csv"
export_path_desc = export_dir / f"{year}_samplesize-desc.tsv"
export_path_plot = export_dir / f"{year}_samplesize-plot.png"

# Creates pandas datetimes for start, end, COVID declaration.
covid_dt = pd.to_datetime(f"{year}-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta("30D")

# Load data.
data = pd.read_csv(import_path, encoding="utf-8")
df = utils.preprocess_subreddit(data)

# Consolidate flair
dream_flair = ["Short Dream", "Medium Dream", "Long Dream"]
nondream_flair = [ f for f in df["link_flair_text"].unique() if f not in dream_flair ]
df["Dream flair"] = (df["link_flair_text"]
    .fillna(value="None")
    .replace(to_replace=nondream_flair, value="None")
)

# Average post counts per day.
daily = (df.
    groupby([pd.Grouper(key="timestamp", freq="D"), "Dream flair"])
    .size()
    .unstack()
    .fillna(0)
    .sort_index(ascending=True)
)
daily["total"] = daily.sum(axis=1)
daily["dream"] = daily[dream_flair].sum(axis=1)

# Extract window of interest.
covid_dt = pd.to_datetime(f"{year}-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta("30D")
daily = daily.loc[start_dt:end_dt, :]
daily["Window"] = pd.Series(daily.index).between(covid_dt, end_dt, inclusive="both").to_numpy()
daily["Window"] = daily["Window"].replace({False: "Pre", True: "Post"})

# Create a dataframe with mean, std, etc. for the number of posts per day.
desc = (daily
    .groupby("Window")
    .agg(["count", "min", "max", "mean", "sum"])
    .stack("Dream flair")
    .sort_index(ascending=False)
    .round(2)
)

# Export descriptives.
desc.to_csv(export_path_desc, sep="\t")


############################################
################  Plotting  ################
############################################

# Set global matplotlib settings.
utils.load_matplotlib_settings()

# Select colors.
colormap = cc.cm.blues
palette = {
    "None": "white",
    "Short Dream": colormap(1/3),
    "Medium Dream": colormap(2/3),
    "Long Dream": colormap(3/3),
}

# Open figure.
fig, ax = plt.subplots(figsize=(3.8, 1.5))

# Identify histogram bins.
lower_xbound = mdates.date2num(start_dt)
upper_xbound = mdates.date2num(end_dt + pd.Timedelta("1D"))
bins = np.arange(lower_xbound, upper_xbound)

# Draw data.
ax = sns.histplot(
    df,
    x="timestamp",
    hue="Dream flair",
    multiple="stack",
    palette=palette,
    hue_order=list(palette),
    bins=bins,
    edgecolor="black",
    linewidth=.5,
    ax=ax,
    clip_on=False,
)

# Adjust aesthetics.
ax.margins(x=0)
ax.set_ybound(upper=190)
ax.set_xlabel(None)
ax.set_ylabel("Daily post count")
ax.tick_params(axis="x", which="both", direction="out", top=False)
ax.spines[["left", "right"]].set_position(("outward", 7))
date_major_locator = mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1)
date_minor_locator = mdates.DayLocator(bymonthday=None, interval=1)
date_major_formatter = mdates.DateFormatter(fr"%B $1^\mathrm{{st}}$, {year}")
ax.xaxis.set_major_locator(date_major_locator)
ax.xaxis.set_minor_locator(date_minor_locator)
ax.xaxis.set_major_formatter(date_major_formatter)
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
sns.move_legend(
    ax,
    "upper center",
    ncol=4,
    borderaxespad=0,
    columnspacing=1,
    handlelength=1,
    handleheight=1,
)

# Export.
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()
