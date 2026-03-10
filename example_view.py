"""
Export a file that merges the nightmare and anxious dreaming scores
of each post, filters to dreams, and subsets the window of interest.
The exported file makes it easier to qualitatively inspect how well
the automated measures perform.
"""

from pathlib import Path

import pandas as pd

import utils

# Declare filepaths for importing and exporting
derivatives_dir = Path(utils.config["derivatives_directory"])
sourcedata_dir = Path(utils.config["sourcedata_directory"])
import_path_raw = sourcedata_dir / "r-dreams.csv"
export_path = derivatives_dir / "example_view.csv"

# Load data
df = utils.read_liwc_csv(subreddit="dreams", dream_filter="dreams")

# Extract the relevant time window
covid_dt = pd.to_datetime("2020-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta("30D")
df = df[df["timestamp"].between(start_dt, end_dt, inclusive="both")]
df = df.drop(columns="timestamp")

# Sort from high-to-low anxiety (can sort nightmare in external software)
df = df.sort_values("anxiety", ascending=False)

# Reorder columns for easier viewing of long text in external software
df = df.reindex(columns=["anxiety", "nightmare", "title", "post"])

# Export
df.to_csv(export_path, index=False, encoding="utf-8")
