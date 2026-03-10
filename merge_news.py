"""
Merge the news 2020 file with news 2019 file.
"""

import pandas as pd

import utils

sourcedata_dir = utils.config["sourcedata_directory"]
derivatives_dir = utils.config["derivatives_directory"]

export_path = derivatives_dir / "r-news.csv"

news2020_filepath = utils.fetch_sourcedata("r-news.csv")
news2019_filepath = utils.fetch_sourcedata("news_since-2019-03-01_until-2019-10-01.jsonl")

news2020 = pd.read_csv(news2020_filepath, encoding="utf-8")
news2019 = pd.read_json(news2019_filepath, lines=True, encoding="utf-8")
# There are some user-specific subreddits in the 2019 data
news2019 = news2019.query("subreddit == 'news'").reset_index(drop=True)

overlapping_columns = news2019.columns.intersection(news2020.columns)

news = pd.concat(
    [news2019[overlapping_columns], news2020[overlapping_columns]], ignore_index=True
)

news.to_csv(export_path, mode="x", index=False, encoding="utf-8")
