# covidreams

A research project using [r/Dreams](https://www.reddit.com/r/Dreams) to look at how the first COVID-19 wave impacted dysphoric dreaming.

## General files

- `environment.yaml` can be used to construct the Python environment
- `config.json` has general parameter options that apply to multiple scripts
- `utils.py` has general functions that are useful to multiple scripts

## Data collection

```bash
# Generate data directory structure.
python setup_directories.py     #> sourcedata/
                                #> derivatives/
                                #> results/

# Scrape posts from r/Dreams and r/news.
python scrape_reddit.py -r Dreams --start 2019-01-01 --end 2020-12-31   #> r-dreams.csv[pkl]
python scrape_reddit.py -r news --start 2019-01-01 --end 2020-12-31     #> r-news.csv[pkl]
```

## Data analysis

```bash
# Run LIWC on raw Reddit data
python liwc_run.py          #> liwc22-dreams-anxiety.csv
                            #> liwc22-dreams-nightmare.csv
                            #> liwc22-news-covid.csv

# Save filtered and sorted file for qualitative inspection
python example_view.py      #> example_view.tsv

# How much data is there?
python samplesize.py        #> 2020_samplesize-desc.tsv
                            #> 2020_samplesize-plot.png
                            
# Dream anxiety time-series interrupted by COVID
python anxiety_regr.py      #> 2020_dreams_anxiety_regr_30-desc.tsv
                            #> 2020_dreams_anxiety_regr_30-stat.txt
                            #> 2020_dreams_anxiety_regr_30-plot.png/pdf

# Correlation between r/Dreams anxiety and COVID r/news
python anxiety_corr.py      #> 2020_dreams_anxiety_corr-desc.tsv
                            #> 2020_dreams_anxiety_corr-stat.tsv
                            #> 2020_dreams_anxiety_corr-plot.png/pdf
                            #> 2020_dreams_anxiety_corr-acor.png/pdf

# Nightmare frequency before and after COVID pandemic announcement
python nightmares_chi2.py   #> 2020_dreams_nightmares_chi2-desc.tsv
                            #> 2020_dreams_nightmares_chi2-stat.tsv
                            #> 2020_dreams_nightmares_chi2-plot.png/pdf

# Run the same analyses using data from 2019 to control for seasonality
python samplesize.py --year 2019
python anxiety_regr.py --year 2019
python anxiety_corr.py --year 2019
python nightmares_chi2.py --year 2019

# Run the same analyses using only non-flaired posts to control for daily language
python anxiety_regr.py --posts wake
python anxiety_corr.py --posts wake
python nightmares_chi2.py --posts wake

# Run the interrupted time series with longer post-COVID time periods
python anxiety_regr.py --year 2020 --posts dreams --days 60
python anxiety_regr.py --year 2020 --posts dreams --days 90

# Run the interrupted time series on nightmare frequency
python anxiety_regr.py --year 2020 --posts dreams --days 60 --category nightmares
python anxiety_regr.py --year 2020 --posts dreams --days 90 --category nightmares
```
