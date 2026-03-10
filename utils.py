"""Helper functions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Load configuration file so it's accessible from utils
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def read_liwc_csv(subreddit, dream_filter=None):
    assert subreddit in {"dreams", "news"}
    if dream_filter is not None:
        assert subreddit == "dreams"
    read_csv_kwargs = dict(index_col="id", encoding="utf-8", parse_dates=["timestamp"])
    import_path = Path(config["derivatives_directory"]) / f"r-{subreddit}-liwc.csv"
    df = pd.read_csv(import_path, **read_csv_kwargs)
    if dream_filter is not None:
        df = filter_dreams(df, dream_filter)
    return df


def filter_dreams(dataframe, filter):
    assert filter in {"dreams", "wake"}
    dream_filter = dataframe["flair"].isin(config["dream_flair"])
    if filter == "dreams":
        df = dataframe[dream_filter]
    elif filter == "wake":
        df = dataframe[~dream_filter]
    return df


def load_matplotlib_settings():
    # plt.rcParams["interactive"] = True
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.cal"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["axes.linewidth"] = 0.8  # edge line width
    plt.rcParams["axes.axisbelow"] = True
    # plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["axes.grid.which"] = "major"
    plt.rcParams["axes.labelpad"] = 4
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["grid.alpha"] = 1
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.title_fontsize"] = 8
    plt.rcParams["legend.borderpad"] = 0.4
    plt.rcParams["legend.labelspacing"] = (
        0.2  # the vertical space between the legend entries
    )
    plt.rcParams["legend.handlelength"] = 2  # the length of the legend lines
    plt.rcParams["legend.handleheight"] = 0.7  # the height of the legend handle
    plt.rcParams["legend.handletextpad"] = (
        0.2  # the space between the legend line and legend text
    )
    plt.rcParams["legend.borderaxespad"] = (
        0.5  # the border between the axes and legend edge
    )
    plt.rcParams["legend.columnspacing"] = (
        1  # the space between the legend line and legend text
    )
