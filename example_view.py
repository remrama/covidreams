"""
Export a file that merges the nightmare and anxious dreaming scores
of each post, filters to dreams, and subsets the window of interest.
The exported file makes it easier to qualitatively inspect how well
the automated measures perform.
"""

import pandas as pd

import utils

# Declare filepaths for importing and exporting
derivatives_dir = utils.config["derivatives_directory"]
sourcedata_dir = utils.config["sourcedata_directory"]
import_path_raw = sourcedata_dir / "r-dreams.csv"
export_path = derivatives_dir / "example_view.csv"

PRE_WINDOW_DURATION = "29D"  # 30 days including event date
POST_WINDOW_DURATION = "30D"
EVENT_DATE = "2020-03-11"
event_dt = pd.to_datetime(EVENT_DATE, utc=False)
start_dt = event_dt - pd.Timedelta(PRE_WINDOW_DURATION)
end_dt = event_dt + pd.Timedelta(POST_WINDOW_DURATION)
start_date = start_dt.date().isoformat()
end_date = end_dt.date().isoformat()

(
    # Load data
    utils.read_liwc_csv(subreddit="dreams", dream_filter="dreams")
    # Extract the relevant time window
    .loc[start_date:end_date]
    # Sort from high-to-low anxiety (can sort nightmare in external software)
    .sort_values("anxiety", ascending=False)
    # Reorder columns for easier viewing of long text in external software
    .reindex(columns=["anxiety", "nightmare", "title", "post"])
).to_csv(export_path, index=False, encoding="utf-8")
