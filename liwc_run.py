"""Run LIWC-22 word count analysis on dreams and news datasets."""

import subprocess
import sys
from pathlib import Path

import utils

sourcedata_dir = Path(utils.config["sourcedata_directory"])
derivatives_dir = Path(utils.config["derivatives_directory"])
dic_filepath = sourcedata_dir / "my.dic"

iterations = [
    ("liwc-dreams-anxiety.csv", "dreams", "selftext", "LIWC22", "WC,emo_anx"),
    ("liwc-dreams-nightmare.csv", "dreams", "title", "custom.dic", "nightmare"),
    ("liwc-news-covid.csv", "news", "title", "custom.dic", "WC,covid"),
]

for keys in iterations:
    export_name, subreddit, column, dictionary, include_categories = keys
    import_path = sourcedata_dir / f"r-{subreddit}.csv"
    export_path = derivatives_dir / export_name
    if dictionary == "custom.dic":
        dictionary = str(dic_filepath)
    if column == "title":
        column_indices = 14
    elif column == "selftext":
        column_indices = 15
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
        "5",
        "--precision",
        "2",
        "--auto-open",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
