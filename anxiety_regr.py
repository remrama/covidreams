"""
Run Interrupted Time Series analysis to see how
the declaration of COVID-19 as a global pandemic
influenced rates of LIWC anxiety words in r/Dreams posts.

Imports 1 file:
    - r/Dreams LIWC output, LIWC22 dictionary run on posts

Exports 4 files:
    - model as a pickle file
    - model data values as a tsv file
    - model stats as a txt file
    - model plot as a png file
    - model plot as a pdf file
"""
import argparse
from pathlib import Path

import colorcet as cc
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import utils


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", default=2020, choices=[2019, 2020], type=int)
parser.add_argument("-p", "--posts", default="dreams", choices=["dreams", "wake"], type=str)
args = parser.parse_args()

year = args.year
posts = args.posts

# Declare filepaths for importing and exporting.
derivatives_dir = Path(utils.config["derivatives_directory"])
import_path = derivatives_dir / "LIWC-22 Results - r-dreams_posts - LIWC Analysis.csv"
export_path_modl = derivatives_dir / f"{year}_{posts}_anxiety_regr-modl.pkl"
export_path_vals = derivatives_dir / f"{year}_{posts}_anxiety_regr-vals.tsv"
export_path_stat = derivatives_dir / f"{year}_{posts}_anxiety_regr-stat.txt"
export_path_plot = derivatives_dir / f"{year}_{posts}_anxiety_regr-plot.png"

# Creates pandas datetimes for start, end, COVID declaration.
covid_dt = pd.to_datetime(f"{year}-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta("30D")

# Load data.
data = pd.read_csv(import_path)
drms = utils.filter_flair(data, posts=posts)
df = utils.preprocess_subreddit(drms)

# Average per day.
daily = (df
    .groupby(pd.Grouper(key="timestamp", freq="D"))
    ["emo_anx"].mean()
    .sort_index(ascending=True)
    .to_frame()
)

# Shift dream anxiety back one day since posts are from dreams occuring the previous day.
daily["emo_anx"] = daily["emo_anx"].shift(-1)

# Get a smoothed version for plotting.
daily_smooth = daily.rolling(window=7,center=True)["emo_anx"].mean()

# Simplify timestamp index as a new date column.
daily["date"] = pd.Series(daily.index.to_frame()["timestamp"])

# Extract the relevant time window.
daily = daily.loc[daily["date"].between(start_dt, end_dt, inclusive="both"), :]

# Add columns for regression.
daily["Time"] = range(1, len(daily) + 1)
daily["Covid"] = daily["date"].gt(covid_dt).astype(int)
daily["TimeCovid"] = daily["Covid"].cumsum()

# Run regression.
model = smf.ols(formula="emo_anx ~ Time + Covid + TimeCovid", data=daily)
model = model.fit()

# Extract measures for plotting and exporting.
summary = model.summary()
observed = model.get_prediction().summary_frame(alpha=0.05)

# Run regression on pre-covid to get counterfactual/predicted line.
daily_precovid = daily.set_index("date").loc[start_dt:covid_dt]
model_precovid = smf.ols(formula="emo_anx ~ Time + Covid + TimeCovid", data=daily_precovid)
model_precovid = model_precovid.fit()
daily_postcovid = daily.set_index("date").loc[covid_dt:]
predicted = model_precovid.predict(daily_postcovid)

# Compile single dataframe with relevant values.
dat = daily["emo_anx"].rename("data")
datsmooth = daily_smooth.rename("datasmooth")
obs = (observed
    .drop(columns=[c for c in observed if "obs" in c ])
    .rename(columns=lambda x: x.replace("mean", "obs"))
)
pred = predicted.rename("pred")
model_vals = obs.join(pred).join(dat).join(datsmooth)

# Export stats.
model.save(export_path_modl)
model_vals.to_csv(export_path_vals, na_rep="NA", sep="\t")
with open(export_path_stat, "w", encoding="utf-8") as f:
    f.write(summary.as_text())


############################################
################  Plotting  ################
############################################

# Set global matplotlib settings.
utils.load_matplotlib_settings()

# Select colors.
colormap = cc.cm.bwy
data_color = colormap(0.)
regr_color = colormap(1.)

# Convert datetimes to x-axis values.
start_x = mdates.date2num(start_dt)
covid_x = mdates.date2num(covid_dt)
end_x = mdates.date2num(end_dt)

# Open the figure.
fig, ax = plt.subplots(figsize=(2.8, 2))

# Draw data.
xvals = daily_smooth.index.to_numpy()
yvals = daily_smooth.to_numpy()
ax.plot(xvals, yvals, c=data_color, lw=1, label="Data")
ax.fill_between(xvals, yvals, color=data_color, lw=0, alpha=0.3)

# Draw regression line.
xvals = observed.index.to_numpy()
yvals = observed["mean"].to_numpy()
evals_lo = observed["mean"].sub(observed["mean_se"]).to_numpy()
evals_hi = observed["mean"].add(observed["mean_se"]).to_numpy()
ax.plot(xvals, yvals, c=regr_color, lw=1, label="Observed")
ax.fill_between(xvals, evals_lo, evals_hi, color=regr_color, lw=0, alpha=0.3)

# Draw predicted regression line.
xvals_ = predicted.index.to_numpy()
yvals_ = predicted.to_numpy()
ax.plot(xvals_, yvals_, c=regr_color, lw=1, ls="dotted", label="Predicted")

# Draw stats results.
beta = model.params.loc["Covid"]
pval = model.pvalues.loc["Covid"]
asterisks = "*" * sum( pval < cutoff for cutoff in [0.05, 0.01, 0.001] )
stat_txt = asterisks + fr"$B$ = {beta:.2f}"
# stat_txt = asterisks + fr"$\beta$ = {b:.2f}"
ax.text(0.95, 0.9, stat_txt, ha="right", va="top", transform=ax.transAxes)

# Draw legend.
ax.legend(loc="lower left")

# Adjust x-axis aesthetics.
ax.set_xbound(lower=start_x, upper=end_x)
date_major_locator = mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1)
date_minor_locator = mdates.WeekdayLocator(byweekday=1, interval=1)
date_major_formatter = mdates.DateFormatter(r"%B $1^\mathrm{st}$")
ax.xaxis.set_major_locator(date_major_locator)
ax.xaxis.set_minor_locator(date_minor_locator)
ax.xaxis.set_major_formatter(date_major_formatter)

# Adjust y-axis aesthetics.
ymin = 0.2
ymax = 0.5
if posts == "wake":
    ymax += 0.1
ax.set_ylim(ymin, ymax)
ax.set_ylabel("Anxious dreaming")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

# Draw COVID-declaration/intervention line.
who_text = fr"March $11^\mathrm{{th}}$, {year}"
if year == 2020:
    who_text += "\nCOVID-19 declared\na global pandemic"
xy = (mdates.date2num(covid_dt), ymin)
xytext = (0.45, 0.9)
xy_coords = "data"
xytext_coords = "axes fraction"
arrowprops = dict(
    arrowstyle="-|>",
    ls="solid",
    color="black",
    connectionstyle="angle,angleA=0,angleB=-90,rad=0",
)
ax.annotate(
    who_text,
    ha="right", va="top",
    xy=xy, xycoords=xy_coords,
    xytext=xytext, textcoords=xytext_coords,
    arrowprops=arrowprops,
)

# Export plots.
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()
