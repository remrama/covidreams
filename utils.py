"""Helper functions.
"""
import json
import matplotlib.pyplot as plt
import pandas as pd


# Load configuration file so it's accessible from utils
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def preprocess_subreddit(df, column="selftext"):
    assert column in ["selftext", "title"]

    # Create proper timestamp column.
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

    # Remove deleted posts, removed posts, and posts without any text.
    df = df[~df[column].isin(["[deleted]", "[removed]"])]
    df = df.dropna(subset=[column])

    # Remove duplicated posts.
    df = df.drop_duplicates(subset=[column], keep="first")

    # Ensure words.
    if "WC" in df:
        df = df.query("WC >= 1")
    else:
        df = df[df[column].str.len().ge(1)]

    return df.reset_index(drop=True)


def filter_flair(df, posts="dreams"):
    assert posts in ["dreams", "wake"]

    # Reduce to dreams only (unless running control).
    dream_flair = ["Short Dream", "Medium Dream", "Long Dream"]
    post_idx = df["link_flair_text"].isin(dream_flair)
    if posts == "wake":
        post_idx = ~post_idx
    df = df.loc[post_idx, :]

    return df.reset_index(drop=True)


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
    plt.rcParams["axes.linewidth"] = 0.8 # edge line width
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
    plt.rcParams["legend.borderpad"] = .4
    plt.rcParams["legend.labelspacing"] = .2 # the vertical space between the legend entries
    plt.rcParams["legend.handlelength"] = 2 # the length of the legend lines
    plt.rcParams["legend.handleheight"] = .7 # the height of the legend handle
    plt.rcParams["legend.handletextpad"] = .2 # the space between the legend line and legend text
    plt.rcParams["legend.borderaxespad"] = .5 # the border between the axes and legend edge
    plt.rcParams["legend.columnspacing"] = 1 # the space between the legend line and legend text
