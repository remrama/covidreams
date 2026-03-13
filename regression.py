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
    - model plot as a svg file
    - autocorrelation check plot and stats as a png file
"""

import argparse

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prioryear", action="store_true", help="Run on 2019 data instead of 2020 data."
)
parser.add_argument(
    "--nondreams", action="store_true", help="Run on wake text instead of dream text."
)
parser.add_argument(
    "--longerwindow", action="store_true", help="Run with 60-day window instead of 30."
)
args = parser.parse_args()

year = 2019 if args.prioryear else 2020
flair = "nondreams" if args.nondreams else "dreams"
days = 60 if args.longerwindow else 30

ylabel = "Anxious dreaming"

# Declare filepaths for importing and exporting
derivatives_dir = utils.config["derivatives_directory"]
export_parent = derivatives_dir
if year == 2019:
    export_parent = export_parent / "prioryear"
if flair == "nondreams":
    export_parent = export_parent / "nondreams"
if days == 60:
    export_parent = export_parent / "longerwindow"
export_parent.mkdir(exist_ok=True)

# Creates pandas datetimes for main COVID event and start, end of windows
PRE_WINDOW_DURATION = "29D" # 30 days including event date
post_window_duration = f"{days:d}D"
event_date = f"{year}-03-11"
covid_dt = pd.to_datetime(event_date, utc=True)
start_dt = covid_dt - pd.Timedelta(PRE_WINDOW_DURATION)
end_dt = covid_dt + pd.Timedelta(post_window_duration)
start_date = start_dt.date().isoformat()
end_date = end_dt.date().isoformat()


def run_regression(flair):
    TEXT_COLUMN = "anxiety"
    export_path = export_parent / "regression-results.pkl"

    # Load data
    df = utils.read_liwc_csv(subreddit="dreams", dream_filter=flair)

    # Get average values per day
    daily = (
        df.resample("1D")[TEXT_COLUMN]
        .mean()
        .sort_index(ascending=True)
        .to_frame()
    )

    # Shift dream anxiety back one day since posts are from dreams occuring the previous day
    daily[TEXT_COLUMN] = daily[TEXT_COLUMN].shift(-1)

    # Save a smoothed version for later access when plotting
    daily[TEXT_COLUMN + "_smooth"] = daily[TEXT_COLUMN].rolling(window=7, center=True).mean()

    # # Simplify timestamp index as a new date column
    # daily["date"] = daily.index.to_frame()["timestamp"].dt.date

    # Extract the relevant time window
    daily = daily.loc[start_date:end_date]

    # Add columns for regression
    daily["Time"] = range(1, len(daily) + 1)
    daily.loc[:, "Covid"] = 1
    daily.loc[start_date:event_date, "Covid"] = 0
    daily["TimeCovid"] = daily["Covid"].cumsum()

    # Run regression
    model_formula = f"{TEXT_COLUMN} ~ Time + Covid + TimeCovid"
    model = sm.formula.ols(formula=model_formula, data=daily)
    result = model.fit()

    # # Extract measures for plotting and exporting
    # summary = result.summary()
    # observed = result.get_prediction().summary_frame(alpha=0.05)

    # # Run regression on pre-covid to get counterfactual/predicted line
    # daily_precovid = daily.set_index("date").loc[start_dt:covid_dt]
    # model_precovid = sm.formula.ols(
    #     formula=f"{TEXT_COLUMN} ~ Time + Covid + TimeCovid", data=daily_precovid
    # )
    # model_precovid = model_precovid.fit()
    # daily_postcovid = daily.set_index("date").loc[covid_dt:]
    # predicted = model_precovid.predict(daily_postcovid)

    # # Get a smoothed version for plotting
    # daily_smooth = daily.rolling(window=7, center=True)[TEXT_COLUMN].mean()

    # # Compile single dataframe with relevant values
    # dat = daily[TEXT_COLUMN].rename("data")
    # datsmooth = daily_smooth.rename("datasmooth")
    # obs = observed.drop(columns=[c for c in observed if "obs" in c]).rename(
    #     columns=lambda x: x.replace("mean", "obs")
    # )
    # pred = predicted.rename("pred")
    # model_vals = obs.join(pred).join(dat).join(datsmooth)

    # # Export 
    result.year = year
    result.save(export_path)
    # model_vals.to_csv(export_path_vals, na_rep="N/A", sep="\t")
    # with open(export_path_stat, "w", encoding="utf-8") as f:
    #     f.write(summary.as_text())
    return result

############################################
################  Plotting  ################
############################################

def plot_regression(result):
    export_path = export_parent / "regression-plot.png"
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

    # Get observed (observed fit) data
    observed = result.get_prediction().summary_frame(alpha=0.05)

    # Open the figure
    fig, ax = plt.subplots(figsize=(2.8, 2))

    # Draw raw data
    raw = result.model.data.frame
    xvals = raw.index.to_numpy()
    yvals = raw["anxiety_smooth"].to_numpy()
    ax.plot(xvals, yvals, c=data_color, lw=1, label="Data")
    ax.fill_between(xvals, yvals, color=data_color, lw=0, alpha=0.3)

    # Draw regression line of observed fit
    observed = result.get_prediction().summary_frame(alpha=0.05)
    xvals = observed.index.to_numpy()
    yvals = observed["mean"].to_numpy()
    evals_lo = observed["mean"].sub(observed["mean_se"]).to_numpy()
    evals_hi = observed["mean"].add(observed["mean_se"]).to_numpy()
    ax.plot(xvals, yvals, c=regr_color, lw=1, label="Observed")
    ax.fill_between(xvals, evals_lo, evals_hi, color=regr_color, lw=0, alpha=0.3)

    # Draw counterfactual regression line
    cf_data = result.model.data.frame.assign(Covid=0).assign(TimeCovid=0)
    cf_result = result.get_prediction(cf_data).summary_frame(alpha=0.05)
    cf_result.index = result.model.data.frame.index
    xvals = cf_result.index.to_numpy()
    yvals = cf_result["mean"].to_numpy()
    ax.plot(xvals, yvals, c=regr_color, lw=1, ls="dotted", label="Predicted")

    # Draw stats results
    beta = result.params.loc["Covid"]
    pval = result.pvalues.loc["Covid"]
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
    if flair == "nondreams":
        ymax += 0.1
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

    # Draw COVID-declaration/intervention line
    who_text = rf"March $11^\mathrm{{th}}$, {result.year}"
    if result.year == 2020:
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
    utils.save_and_close_fig(export_path, include_svg=True)
    return

##########################################################################
################  Stats and Plotting for Autocorrelation  ################
##########################################################################

def autocorrelation(result):
    export_path_stats = export_parent / "regression-acor.tsv"
    export_path_plot = export_path_stats.with_suffix(".png")

    N_LAGS_LB = 10
    N_LAGS_BG = 2
    acf_kwargs = dict(
        alpha=0.05,
        zero=True,
        missing="drop",
        title=None,
        bartlett_confint=False,
        clip_on=False,
    )

    # Durbin-Watson test
    db_stat = sm.stats.durbin_watson(result.resid)
    # Ljung-Box Q-test
    lb = sm.stats.acorr_ljungbox(result.resid, lags=N_LAGS_LB)
    lb_stat, lb_pval = lb.iloc[-1]
    # Breusch-Godfrey test
    bg_test = sm.stats.acorr_breusch_godfrey(result, nlags=N_LAGS_BG)
    bg_stat, bg_pval = map(float, bg_test[:2])
    records = [
        {"test": "Durbin-Watson", "stat": db_stat},
        {"test": "Ljung-Box", "stat": lb_stat, "pval": lb_pval, "nlags": N_LAGS_LB},
        {"test": "Breusch-Godfrey", "stat": bg_stat, "pval": bg_pval, "nlags": N_LAGS_BG},
    ]
    df = pd.DataFrame.from_records(records, index="test").astype({"nlags": "Int64"})
    df["stat"] = df["stat"].round(2)
    df["pval"] = df["pval"].round(5)

    # ACF plot, visual inspection
    strings = [
        f"Durbin-Watson = {db_stat:.2f}",
        f"Ljung-Box (lag={N_LAGS_LB}) = {lb_stat:.1f}, p = {lb_pval:.3f}",
        f"Breusch-Godfrey (lag={N_LAGS_BG}) = {bg_stat:.1f}, p = {bg_pval:.3f}",
    ]
    strings = [
        s.replace("p = 0.", "p = .").replace("p = .000", "p < .001") for s in strings
    ]
    text = "\n".join(strings)
    text_pass = "\n".join(
        [
            "PASS" if 1.5 <= db_stat <= 2.5 else "FAIL",
            "PASS" if lb_pval > 0.05 else "FAIL",
            "PASS" if bg_pval > 0.05 else "FAIL",
        ]
    )
    # Open up figure
    fig, ax = plt.subplots(
        figsize=(3, 3), constrained_layout=True, sharex=True, sharey=True
    )
    # Draw an ACF plot/correlogram to visually inspect autocorrelation
    sm.graphics.tsa.plot_acf(result.resid, ax, **acf_kwargs)
    # Draw text
    title = "Autocorrelation of model residuals"
    ax.text(0.5, 0.95, title, ha="center", va="top", weight="bold", transform=ax.transAxes)
    ax.text(0.83, 0.05, text, ha="right", va="bottom", transform=ax.transAxes)
    ax.text(0.85, 0.05, text_pass, ha="left", va="bottom", transform=ax.transAxes)

    # Export
    df.to_csv(export_path_stats, sep="\t", encoding="utf-8", na_rep="N/A")
    utils.save_and_close_fig(export_path_plot)
    return


if __name__ == "__main__":
    result = run_regression(flair)
    plot_regression(result)
    autocorrelation(result)