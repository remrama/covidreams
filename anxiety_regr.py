"""
Run Interrupted Time Series analysis to see how
the declaration of COVID-19 as a global pandemic
influenced language in r/Dreams posts.

Imports 1 file:
    - r/Dreams LIWC output

Exports 4 files:
    - model as a pickle file
    - model data values as a tsv file
    - model stats as a txt file
    - model plot as a png file
    - model plot as a pdf file
    - autocorrelation check plot and stats as a png file
    - autocorrelation check plot and stats as a pdf file
"""

import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", default=2020, choices=[2019, 2020], type=int)
parser.add_argument(
    "-p", "--posts", default="dreams", choices=["dreams", "wake"], type=str
)
parser.add_argument("-d", "--days", default=30, type=int)
parser.add_argument(
    "-c", "--category", default="anxiety", choices=["anxiety", "nightmares"], type=str
)
args = parser.parse_args()

year = args.year
posts = args.posts
days = args.days
category = args.category

if category == "anxiety":
    text_column = "emo_anx"
    ylabel = "Anxious dreaming"
elif category == "nightmares":
    text_column = "nightmare"
    ylabel = "Nightmare frequency"

# Declare filepaths for importing and exporting
derivatives_dir = Path(utils.config["derivatives_directory"])
export_parent = derivatives_dir / year
export_parent.mkdir(exist_ok=True)
export_path_modl = export_parent / f"regression-modl_{posts}_{category}.pkl"
export_path_vals = export_parent / f"regression-vals_{posts}_{category}.tsv"
export_path_stat = export_parent / f"regression-stat_{posts}_{category}.txt"
export_path_plot = export_parent / f"regression-plot_{posts}_{category}.png"
export_path_acor = export_parent / f"regression-acor_{posts}_{category}.png"

# Creates pandas datetimes for start, end, COVID declaration
covid_dt = pd.to_datetime(f"{year}-03-11", utc=True)
start_dt = covid_dt - pd.Timedelta("30D")
end_dt = covid_dt + pd.Timedelta(f"{days:d}D")

# Load data
data = utils.load_liwc_results(subreddit="dreams")
drms = utils.filter_flair(data, posts=posts)
df = utils.preprocess_subreddit(drms)

# Average per day
daily = (
    df.groupby(pd.Grouper(key="timestamp", freq="D"))[text_column]
    .mean()
    .sort_index(ascending=True)
    .to_frame()
)

# Shift dreams back one day since posts are from dreams occuring the previous day
daily[text_column] = daily[text_column].shift(-1)

# Get a smoothed version for plotting
daily_smooth = daily.rolling(window=7, center=True)[text_column].mean()

# Simplify timestamp index as a new date column
daily["date"] = pd.Series(daily.index.to_frame()["timestamp"])

# Extract the relevant time window
daily = daily.loc[daily["date"].between(start_dt, end_dt, inclusive="both"), :]

# Add columns for regression
daily["Time"] = range(1, len(daily) + 1)
daily["Covid"] = daily["date"].gt(covid_dt).astype(int)
daily["TimeCovid"] = daily["Covid"].cumsum()

# # Extract pre-intervention data for inspecting autocorrelation and stationarity
# preintervention_data = daily.loc[:covid_dt, text_column].to_numpy()

# # Test pre-intervention data for autocorrelation
# dw_stat = sm.stats.durbin_watson(preintervention_data)  # Durbin-Watson test
# lb_stat, lb_p = sm.stats.acorr_ljungbox(preintervention_data, lags=1, return_df=True)  # Ljung-Box Q-test
# lm_stat, lm_p, f_stat, f_p = sm.stats.acorr_breusch_godfrey(model, nlags=2)  # Breusch-Godfrey test
# pass_lb = True if lb_p > 0.05 else False
# pass_bf = True if lm_p > 0.05 else False
# sm.graphics.tsa.plot_acf(preintervention_data, lags=40)  # ACF plot, visual inspection

# # Test pre-intervention data for stationarity
# adf_stat, adf_p, _, _, _, _ = sm.tsa.adfuller(preintervention_data, regression="c", autolag="AIC")  # Augmented Dickey-Fuller test
# kpss_stat, kpss_p, _, _ = sm.tsa.kpss(preintervention_data)  # Kwiatkowski-Phillips-Schmidt-Shin test
# pass_adf = True if adf_p < 0.05 else False
# pass_kpss = True if kpss_p > 0.05 else False

# def inspect_residuals(model, acf_plot=True):
# """Return results from autocorrelation and stationarity tests of residuals."""
# # Breusch-Godfrey test for autocorrelation.
# lm_stat, lm_p, f_stat, f_p = sm.stats.acorr_breusch_godfrey(model, nlags=2)
# # Durbin-Watson test for autocorrelation.
# dw_stat = sm.stats.durbin_watson(model.resid)
# # Ljung-Box Q-test for autocorrelation.
# lb_stat, lb_p = sm.stats.acorr_ljungbox(model.resid, lags=1, return_df=True)
# # Augmented Dickey-Fuller test for stationarity.
# adf_stat, adf_p, _, _, _, _ = sm.tsa.adfuller(preintervention_data, regression="c", autolag="AIC")
# # Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
# kpss_stat, kpss_p, _, _ = sm.tsa.kpss(preintervention_data)
# # Compile into a single dataframe.
# results = pd.DataFrame(
#     {
#         "data": ["residuals", "residuals", "residuals"],
#         "measure": ["autocorrelation", "autocorrelation", "autocorrelation"],
#         "test": ["breusch_godfrey", "ljungbox", "durbin_watson"],
#         "stat": [lm_stat, lb_stat, dw_stat],
#         "pval": [lm_p, lb_p, np.nan],
#     }
# )
# model.model.endog[model.model.exog[:,2]==0]

# Run regression
model = sm.formula.ols(formula=f"{text_column} ~ Time + Covid + TimeCovid", data=daily)
model = model.fit()

# Extract measures for plotting and exporting
summary = model.summary()
observed = model.get_prediction().summary_frame(alpha=0.05)

# Run regression on pre-covid to get counterfactual/predicted line
daily_precovid = daily.set_index("date").loc[start_dt:covid_dt]
model_precovid = sm.formula.ols(
    formula=f"{text_column} ~ Time + Covid + TimeCovid", data=daily_precovid
)
model_precovid = model_precovid.fit()
daily_postcovid = daily.set_index("date").loc[covid_dt:]
predicted = model_precovid.predict(daily_postcovid)

# Compile single dataframe with relevant values
dat = daily[text_column].rename("data")
datsmooth = daily_smooth.rename("datasmooth")
obs = observed.drop(columns=[c for c in observed if "obs" in c]).rename(
    columns=lambda x: x.replace("mean", "obs")
)
pred = predicted.rename("pred")
model_vals = obs.join(pred).join(dat).join(datsmooth)

# Export stats
model.save(export_path_modl)
model_vals.to_csv(export_path_vals, na_rep="NA", sep="\t")
with open(export_path_stat, "w", encoding="utf-8") as f:
    f.write(summary.as_text())


############################################
################  Plotting  ################
############################################

# Set global matplotlib settings
utils.load_matplotlib_settings()

# Select colors
colormap = cc.cm.bwy
data_color = colormap(0.0)
regr_color = colormap(1.0)

# Convert datetimes to x-axis values
start_x = mdates.date2num(start_dt)
covid_x = mdates.date2num(covid_dt)
end_x = mdates.date2num(end_dt)

# Open the figure
fig, ax = plt.subplots(figsize=(2.8, 2))

# Draw data
xvals = daily_smooth.index.to_numpy()
yvals = daily_smooth.to_numpy()
ax.plot(xvals, yvals, c=data_color, lw=1, label="Data")
ax.fill_between(xvals, yvals, color=data_color, lw=0, alpha=0.3)

# Draw regression line
xvals = observed.index.to_numpy()
yvals = observed["mean"].to_numpy()
evals_lo = observed["mean"].sub(observed["mean_se"]).to_numpy()
evals_hi = observed["mean"].add(observed["mean_se"]).to_numpy()
ax.plot(xvals, yvals, c=regr_color, lw=1, label="Observed")
ax.fill_between(xvals, evals_lo, evals_hi, color=regr_color, lw=0, alpha=0.3)

# Draw predicted regression line
xvals_ = predicted.index.to_numpy()
yvals_ = predicted.to_numpy()
ax.plot(xvals_, yvals_, c=regr_color, lw=1, ls="dotted", label="Predicted")

# Draw stats results
beta = model.params.loc["Covid"]
pval = model.pvalues.loc["Covid"]
asterisks = "*" * sum(pval < cutoff for cutoff in [0.05, 0.01, 0.001])
stat_txt = asterisks + rf"$B$ = {beta:.2f}"
# stat_txt = asterisks + fr"$\beta$ = {b:.2f}"
ax.text(0.95, 0.9, stat_txt, ha="right", va="top", transform=ax.transAxes)

# Draw legend
ax.legend(loc="lower left")

# Adjust x-axis aesthetics
ax.set_xbound(lower=start_x, upper=end_x)
date_major_locator = mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1)
date_minor_locator = mdates.WeekdayLocator(byweekday=1, interval=1)
date_major_formatter = mdates.DateFormatter(r"%B $1^\mathrm{st}$")
ax.xaxis.set_major_locator(date_major_locator)
ax.xaxis.set_minor_locator(date_minor_locator)
ax.xaxis.set_major_formatter(date_major_formatter)

# Adjust y-axis aesthetics
ymin = 0.2
ymax = 0.5
if posts == "wake":
    ymax += 0.1
ax.set_ylim(ymin, ymax)
ax.set_ylabel(ylabel)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

# Draw COVID-declaration/intervention line
who_text = rf"March $11^\mathrm{{th}}$, {year}"
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
    ha="right",
    va="top",
    xy=xy,
    xycoords=xy_coords,
    xytext=xytext,
    textcoords=xytext_coords,
    arrowprops=arrowprops,
)

# Export plots
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()


#######################################################################################
################  Stats and Plotting for Autocorrelation/Stationarity  ################
#######################################################################################
# Check autocorrelation and stationarity in residuals.


# Open up figure
fig, ax = plt.subplots(
    figsize=(3, 3), constrained_layout=True, sharex=True, sharey=True
)
# Select universal plotting keyword arguments
data = model.resid
# Extract pre-intervention data for inspecting autocorrelation and stationarity.
# data = daily.loc[:covid_dt, text_column].to_numpy()
data = model.model.endog[model.model.exog[:, 2] == 0]
dw_stat = sm.stats.durbin_watson(data)  # Durbin-Watson test
ljb = sm.stats.acorr_ljungbox(data, lags=1)  # Ljung-Box Q-test
lb_stat = ljb.at[1, "lb_stat"]
lb_p = ljb.at[1, "lb_pvalue"]
lm_stat, lm_p, f_stat, f_p = sm.stats.acorr_breusch_godfrey(
    model, nlags=2
)  # Breusch-Godfrey test
adf_stat, adf_p, _, _, _, _ = sm.tsa.adfuller(
    data, regression="c", autolag="AIC"
)  # Augmented Dickey-Fuller test
kpss_stat, kpss_p, _, _ = sm.tsa.kpss(data)  # Kwiatkowski-Phillips-Schmidt-Shin test
# ACF plot, visual inspection
acf_kwargs = dict(
    alpha=0.05,
    zero=True,
    missing="drop",
    title=None,
    bartlett_confint=False,
    clip_on=False,
)
strings = [
    f"Durbin-Watson = {dw_stat:.2f}",
    f"Ljung-Box = {lb_stat:.1f}, p = {lb_p:.3f}",
    f"Breusch-Godfrey = {lm_stat:.1f}, p = {lm_p:.3f}",
    f"Dickey-Fuller = {adf_stat:.1f}, p = {adf_p:.3f}",
    f"KPSS = {kpss_stat:.1f}, p = {kpss_p:.3f}",
]
strings = [
    s.replace("p = 0.", "p = .").replace("p = .000", "p < .001") for s in strings
]
text = "\n".join(strings)
text_pass = "\n".join(
    [
        "PASS" if 1 < dw_stat < 3 else "FAIL",
        "PASS" if lb_p > 0.05 else "FAIL",
        "PASS" if lm_p > 0.05 else "FAIL",
        "PASS" if adf_p < 0.05 else "FAIL",
        "PASS" if kpss_p > 0.05 else "FAIL",
    ]
)
# Draw an ACF plot/correlogram to visually inspect autocorrelation
sm.graphics.tsa.plot_acf(data, ax, **acf_kwargs)
# Draw text
title = "Autocorrelation and stationarity\nin model residuals"
ax.text(0.5, 0.95, title, ha="center", va="top", weight="bold", transform=ax.transAxes)
ax.text(0.83, 0.05, text, ha="right", va="bottom", transform=ax.transAxes)
ax.text(0.85, 0.05, text_pass, ha="left", va="bottom", transform=ax.transAxes)

# Export plots
plt.savefig(export_path_acor)
plt.savefig(export_path_acor.with_suffix(".pdf"))
plt.close()
