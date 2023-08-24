"""Scrape a given subreddit."""

import argparse
import datetime
from pathlib import Path

import pandas as pd
from psaw import PushshiftAPI
import tqdm

import utils


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--subreddit", required=True, type=str)
parser.add_argument("-s", "--start", required=True, type=str)
parser.add_argument("-e", "--end", required=True, type=str)
args = parser.parse_args()

subreddit = args.subreddit
start_date = args.start
end_date = args.end

# Generate a timestamped filename for exporting.
export_dir = Path(utils.config["source_directory"])
export_basename = f"r-{subreddit}.csv".lower()
export_path = export_dir / export_basename
export_path_pickle = export_path.with_suffix(".pkl")

# Convert timestamp strings to epoch-based integers.
start_epoch = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
end_epoch = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp())

submission_filters = [
    "created_utc",
    "subreddit",
    "subreddit_id",
    "id",
    "author",
    "author_fullname",
    "link_flair_text",
    "num_comments",
    "total_awards_received",
    "score",
    "upvote_ratio",
    "is_video",
    "title",
    "selftext",
]

# Find all relevant Reddit posts.
api = PushshiftAPI()
gen = api.search_submissions(
    subreddit=subreddit,
    filter=submission_filters,
    after=start_epoch,
    before=end_epoch,
)

# Grab all the posts and turn into a dataframe (takes a while).
df = pd.DataFrame([ thing.d_ for thing in tqdm.tqdm(gen, desc=f"Scraping r/{subreddit}") ])

# Reorder columns, including the "created" stamp.
df = df[["created"] + submission_filters]

# Export as a csv file.
df.to_csv(export_path, encoding="utf-8-sig", index=False, na_rep="NA")

# Export as a pickle file as backup.
df.to_pickle(export_path_pickle)
