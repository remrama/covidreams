"""Run LIWC-22 word count analysis on dreams and news datasets."""

import subprocess
import sys

import utils

sourcedata_dir = utils.config["sourcedata_directory"]
derivatives_dir = utils.config["derivatives_directory"]
dic_filepath = utils.fetch_sourcedata("custom.dic")

iterations = [
    ("liwc-dreams-anxiety.csv", "dreams", "selftext", "LIWC22", "emo_anx"),
    ("liwc-dreams-nightmare.csv", "dreams", "title", "custom.dic", "nightmare"),
    ("liwc-news-covid.csv", "news", "title", "custom.dic", "covid"),
]

for keys in iterations:
    export_name, subreddit, column, dictionary, include_categories = keys
    if subreddit == "news":
        import_path = derivatives_dir / f"r-{subreddit}.csv"
    else:
        import_path = utils.fetch_sourcedata(f"r-{subreddit}.csv")
    export_path = derivatives_dir / export_name
    if dictionary == "custom.dic":
        dictionary = str(dic_filepath)
    if column == "title":
        column_indices = 12 if subreddit == "news" else 14
    elif column == "selftext":
        column_indices = 9 if subreddit == "news" else 15
    row_id_indices = 4 if subreddit == "news" else 5
    cmd = [
        sys.executable,
        "liwc.py",
        "wc",
        "--input",
        str(import_path),
        "--output",
        str(export_path),
        "--include-categories",
        include_categories,
        "--column-indices",
        str(column_indices),
        "--dictionary",
        dictionary,
        "--row-id-indices",
        str(row_id_indices),
        "--precision",
        "2",
        "--auto-open",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
