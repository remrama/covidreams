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

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prioryear", action="store_true", help="Run on 2019 data instead of 2020 data."
)
args = parser.parse_args()

year = 2019 if args.prioryear else 2020

# Declare filepaths for importing/exporting
import_dir = utils.config["sourcedata_directory"]
export_dir = utils.config["derivatives_directory"]
if year == 2019:
    export_dir = export_dir / "prioryear"
export_dir.mkdir(exist_ok=True)

# Extract window of interest
PRE_WINDOW_DURATION = "29D"
POST_WINDOW_DURATION = "30D"
event_date = f"{year}-03-11"
event_dt = pd.to_datetime(event_date, utc=False)
start_dt = event_dt - pd.Timedelta(PRE_WINDOW_DURATION)
end_dt = event_dt + pd.Timedelta(POST_WINDOW_DURATION)
start_date = start_dt.date().isoformat()
end_date = end_dt.date().isoformat()


def get_table(dataframe):
    export_path = export_dir / "samplesize-desc.tsv"

    df = dataframe.loc[start_date:end_date]

    # Average post counts per day
    daily = (
        df.reset_index(drop=False)
        .groupby([pd.Grouper(key="timestamp", freq="D"), "flair"])
        .size()
        .unstack()
        .fillna(0)
        .sort_index(ascending=True)
        .astype(int)
    )
    # daily["total"] = daily.sum(axis=1)
    # daily["dream"] = daily[utils.config["dream_flair"]].sum(axis=1)

    # daily = daily.loc[start_date:end_date]
    daily.loc[:, "PostCovid"] = True
    daily.loc[start_date:event_date, "PostCovid"] = False

    # Create a dataframe with mean, std, etc. for the number of posts per day
    desc = (
        daily.groupby("PostCovid")
        .agg(["count", "min", "max", "mean", "sum"])
        .stack("flair", future_stack=True)
        .sort_index(ascending=False)
        .round(2)
        .rename(
            columns={
                "count": "n_days",
                "sum": "n_posts",
            }
        )
    )

    desc.to_csv(export_path, sep="\t", encoding="utf-8", float_format="{:.1f}")
    return


############################################
################  Plotting  ################
############################################


def plot_samplesize(dataframe):
    export_path = export_dir / "samplesize-plot.png"
    # Set global matplotlib settings
    utils.load_matplotlib_settings()

    dataframe = dataframe.rename(columns={"flair": "Dream flair"})

    # Select colors
    colormap = cc.cm.blues
    palette = {
        "None": "white",
        "Short Dream": colormap(1 / 3),
        "Medium Dream": colormap(2 / 3),
        "Long Dream": colormap(3 / 3),
    }

    # Open figure
    fig, ax = plt.subplots(figsize=(3.8, 1.5))

    # Identify histogram bins
    lower_xbound = mdates.date2num(start_dt)
    upper_xbound = mdates.date2num(end_dt + pd.Timedelta("1D"))
    bins = np.arange(lower_xbound, upper_xbound)

    # Draw data
    ax = sns.histplot(
        dataframe.reset_index(drop=False),
        x="timestamp",
        hue="Dream flair",
        multiple="stack",
        palette=palette,
        hue_order=list(palette),
        bins=bins,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        clip_on=False,
    )

    # Adjust aesthetics
    ax.margins(x=0)
    ax.set_ybound(upper=190)
    ax.set_xlabel(None)
    ax.set_ylabel("Daily post count")
    ax.tick_params(axis="x", which="both", direction="out", top=False)
    ax.spines[["left", "right"]].set_position(("outward", 7))
    date_major_locator = mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1)
    date_minor_locator = mdates.DayLocator(bymonthday=None, interval=1)
    date_major_formatter = mdates.DateFormatter(rf"%B $1^\mathrm{{st}}$, {year}")
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

    # Export
    utils.save_and_close_fig(export_path, include_svg=True)


if __name__ == "__main__":
    # Load data
    df = utils.read_liwc_csv(subreddit="dreams")

    # Consolidate flair
    df["flair"] = df["flair"].map(
        lambda x: x if x in utils.config["dream_flair"] else "None"
    )

    get_table(df)
    plot_samplesize(df)
