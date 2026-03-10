# covidreams

A research project using [r/Dreams](https://www.reddit.com/r/Dreams) to look at how the first COVID-19 wave impacted dysphoric dreaming.

## General files

- `environment.yaml` can be used to construct the Python environment
- `config.json` has general parameter options that apply to multiple scripts
- `utils.py` has general functions that are useful to multiple scripts
- `runall.py` can be used to run all scripts in order

## Data collection

```bash
# Scrape posts from r/Dreams and r/news
python scrape_reddit.py -r Dreams --start 2019-01-01 --end 2020-12-31
python scrape_reddit.py -r news --start 2019-01-01 --end 2020-12-31
```

## Data analysis

```bash
# Run LIWC on raw Reddit data
python liwc_run.py
# Merge LIWC results files into easily-accessible files for each subreddit
python merge_liwc.py
# Save filtered and sorted file for qualitative inspection
python example_view.py
# How much data is there?
python samplesize.py
# Dream anxiety time-series interrupted by COVID
python regression.py
# Nightmare frequency before and after COVID pandemic announcement
python chisquared.py
# Correlation between r/Dreams anxiety and COVID r/news
python correlation.py

# Run the same analyses using data from 2019 to control for seasonality
python samplesize.py --prioryear
python regression.py --prioryear
python chisquared.py --prioryear
python correlation.py --prioryear

# Run the same analyses using only non-flaired posts to control for daily language
python regression.py --nondreams
python chisquared.py --nondreams
python correlation.py --nondreams

# Run the interrupted time series with longer post-COVID time period
python regression.py --longerwindow
```
