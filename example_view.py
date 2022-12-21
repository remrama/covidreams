"""
Export a file that merges the nightmare and anxious dreaming scores
of each post, filters to dreams, and subsets the window of interest.
The exported file makes it easier to qualitatively inspect how well
the automated measures perform.

Imports 2 files:
    - r/Dreams LIWC output, LIWC22 dictionary run on posts
    - r/Dreams LIWC output, LIWC22 dictionary run on titles

Exports 1 file:
    - merged and filtered tsv file
"""
import argparse
from pathlib import Path

import pandas as pd

import utils


# Declare filepaths for importing and exporting.
derivatives_dir = Path(utils.config["derivatives_directory"])
import_name_titles = "LIWC-22 Results - r-dreams_titles - LIWC Analysis.csv"
import_path_posts = derivatives_dir / "LIWC-22 Results - r-dreams_posts - LIWC Analysis.csv"
import_path_titles = derivatives_dir / "LIWC-22 Results - r-dreams_titles - LIWC Analysis.csv"
export_path = derivatives_dir / "example_view.tsv"

# Load data.
data_posts = pd.read_csv(import_path_posts)
data_titles = pd.read_csv(import_path_titles)
posts = utils.filter_flair(data_posts)
titles = utils.filter_flair(data_titles)
df_posts = utils.preprocess_subreddit(posts, column="selftext")
df_titles = utils.preprocess_subreddit(titles, column="title")

# Reduce to only desired columns and merge.
df_posts = df_posts.set_index("id")[["timestamp", "title", "selftext", "emo_anx"]]
df_titles = df_titles.set_index("id")[["nightmare"]]
df = df_posts.join(df_titles)

# Extract the relevant time window.
covid_dt = pd.to_datetime("2020-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta("30D")
df = df[df["timestamp"].between(start_dt, end_dt, inclusive="both")]
df = df.drop(columns="timestamp")

# Sort from high-to-low anxiety (can sort nightmare in external software).
df = df.sort_values("emo_anx", ascending=False)

# Reorder columns for easier viewing of long test in external software).
df = df[["emo_anx", "nightmare", "title", "selftext"]]

# Export.
df.to_csv(export_path, index=False, sep="\t")
