"""
Merge LIWC results with raw data, preprocess/filter, and consolidate flair.
"""

import pandas as pd

import utils

sourcedata_dir = utils.config["sourcedata_directory"]
derivatives_dir = utils.config["derivatives_directory"]


def read_raw_reddit(filepath):
    """Reads output from scrape_reddit.py"""
    df = pd.read_csv(filepath, index_col="id", encoding="utf-8")
    return df


def read_liwc_output(filepath):
    """Drops unused Segment column and renames/sets Index on custom LIWC file"""
    df = (
        pd.read_csv(filepath, index_col="Row ID")
        .drop(columns=["Segment"])
        .rename_axis("id")
    )
    return df


def filter_rows(dataframe):
    df = (
        dataframe.query("selftext != '[deleted]'")
        .query("title != '[deleted]'")
        .query("selftext != '[removed]'")
        .query("title != '[removed]'")
        .dropna(subset=["title"], axis="index")
    )
    if "news" in dataframe["subreddit"].tolist():
        # Drop duplicate titles in r/news since they might be the same links
        # but keep duplicate r/Dreams titles since they might be the same with different bodies
        df = df.drop_duplicates(subset=["title"], keep="first")
    if "Dreams" in dataframe["subreddit"].tolist():
        # Almost all r/news posts have empty bodies because they are links
        df = df.dropna(subset=["selftext"], axis="index")
    if "Dreams" in dataframe["subreddit"].tolist():
        df = df.query("selftext.str.split().str.len() >= 10", engine="python")
    if "news" in dataframe["subreddit"].tolist():
        df = df.query("title.str.split().str.len() >= 3", engine="python")
    return df


def filter_columns(dataframe):
    drop_columns = [
        "author",
        "author_fullname",
        "is_video",
        "num_comments",
        "score",
        "subreddit",
        "subreddit_id",
        "total_awards_received",
    ]
    if "Dreams" in dataframe["subreddit"].tolist():
        # Two columns that are only in the r/Dreams output
        drop_columns += ["created", "upvote_ratio"]
    if "news" in dataframe["subreddit"].tolist():
        # Not using selftext for r/news since mostly empty, and flair also useless
        drop_columns += ["link_flair_text", "selftext"]
    rename_columns = {
        "created_utc": "timestamp",
        "link_flair_text": "flair",
        "selftext": "post",
        "emo_anx": "anxiety",
    }
    df = dataframe.drop(columns=drop_columns).rename(columns=rename_columns)
    # Convert created UTC from unix to ISO timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


# Raw r/news comes from derivatives folder bc it required merging source files
dreams_raw = read_raw_reddit(sourcedata_dir / "r-dreams.csv")
news_raw = read_raw_reddit(derivatives_dir / "r-news.csv")

# LIWC results for r/dreams needs to load and merge two separate LIWC outputs
# because two different dictionary files were used. r/news is just one output file.
dreams_liwc_liwc22 = read_liwc_output(derivatives_dir / "liwc-dreams-anxiety.csv")
dreams_liwc_custom = read_liwc_output(derivatives_dir / "liwc-dreams-nightmare.csv")
dreams_liwc = dreams_liwc_liwc22.join(dreams_liwc_custom, how="inner", validate="1:1")
news_liwc = read_liwc_output(derivatives_dir / "liwc-news-covid.csv")

# Merge raw columns with liwc columns
dreams_df = dreams_raw.join(dreams_liwc, how="inner", validate="1:1")
news_df = news_raw.join(news_liwc, how="inner", validate="1:1")

# Filter rows
dreams_df = filter_rows(dreams_df)
news_df = filter_rows(news_df)

# Filter columns
dreams_df = filter_columns(dreams_df)
news_df = filter_columns(news_df)

# Binarize LIWC category columns
dreams_df["nightmare"] = dreams_df["nightmare"].gt(0).astype(int)
news_df["covid"] = news_df["covid"].gt(0).astype(int)

# Save merged and filtered dataframes to derivatives folder
to_csv_kwargs = {
    "index": True,
    "na_rep": "N/A",
    "encoding": "utf-8",
    "lineterminator": "\n",
}

dreams_df.to_csv(derivatives_dir / "r-dreams-liwc.csv", **to_csv_kwargs)
news_df.to_csv(derivatives_dir / "r-news-liwc.csv", **to_csv_kwargs)
