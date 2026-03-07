"""Run LIWC-22 word count analysis on dreams and news datasets."""

import subprocess
import sys
from pathlib import Path

import utils

sourcedata_dir = Path(utils.config["sourcedata_directory"])
derivatives_dir = Path(utils.config["derivatives_directory"])
dic_filepath = sourcedata_dir / "custom.dic"

for subreddit in {"dreams", "news"}:
    import_path = sourcedata_dir / f"r-{subreddit}.csv"
    export_path = derivatives_dir / f"liwc22-{subreddit}.csv"
    cmd = [
        sys.executable,
        "liwc.py",
        "wc",
        "--input",
        str(import_path),
        "--output",
        str(export_path),
        "--include-categories",
        "WC,negemo,covid,nightmare",
        "--combine-columns",
        "no",
        "--column-indices",
        "14,15",
        "--dictionary",
        str(dic_filepath),
        "--output-format",
        "csv",
        "--row-id-indices",
        "5",
        "--precision",
        "2",
        "--threads",
        "-1",
        "--auto-open",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
