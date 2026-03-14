"""
Correlate the frequency of COVID-19-related news headlines
on r/news with the frequency of LIWC anxiety words in dream
reports on r/Dreams.

Inputs 2 files:
    - r/news LIWC output, custom COVID dictionary run on post titles
    - r/Dreams LIWC output, LIWC22 dictionary run on posts

Exports 4 files:
    - values being correlated as a tsv file
    - correlation stats as a tsv file
    - correlation plot as a png file
    - correlation plot as a svg file
    - autocorrelation check plot and stats as a png file
"""

import argparse

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prioryear", action="store_true", help="Run on 2019 data instead of 2020 data."
)
parser.add_argument(
    "--nondreams", action="store_true", help="Run on wake text instead of dream text."
)
args = parser.parse_args()

year = 2019 if args.prioryear else 2020
flair = "nondreams" if args.nondreams else "dreams"

# Declare filepaths for importing/exporting
derivatives_dir = utils.config["derivatives_directory"]
export_parent = derivatives_dir
if year == 2019:
    export_parent = export_parent / "prioryear"
if flair == "nondreams":
    export_parent = export_parent / "nondreams"
export_parent.mkdir(exist_ok=True)

# Choose desired windows
# Have to start post-covid announcement bc otherwise there are crazy outlier jumps
# where the amount of COVID news covid news goes from like .0001 to a lot.
start_date = f"{year}-03-12"
end_date = f"{year}-09-01"
start_dt = pd.to_datetime(start_date, utc=False)
end_dt = pd.to_datetime(end_date, utc=False)


def run_correlation(flair):
    export_path = export_parent / "correlation-stat.tsv"
    # Load data
    drms = utils.read_liwc_csv(subreddit="dreams", dream_filter=flair)
    news = utils.read_liwc_csv(subreddit="news")
    drms["subreddit"] = "Dreams"
    news["subreddit"] = "news"

    # Merge dataframes
    df = pd.concat(
        [drms.reset_index(drop=False), news.reset_index(drop=False)], ignore_index=True
    ).set_index("timestamp").sort_index()

    # Reduce to desired window
    df = df.loc[start_date:end_date]

    # Get weekly averages
    # (Use weekly averages bc otherwise nightmare frequency has many zeros and pct change breaks)
    weekly = (
        df
        .reset_index(drop=False)
        .groupby(["subreddit", pd.Grouper(key="timestamp", freq="W")])[
            ["covid", "anxiety"]
        ]
        .mean()
        .sort_index(ascending=True)
        .unstack(level=0)
        .dropna(axis=1)
        .dropna(axis=0)
        .droplevel(axis=1, level=0)
    )

    # Shift weekly dreams forward to look at how news predicts dreams the following week
    weekly["nextDreams"] = weekly["Dreams"].shift(1)

    # Get percent change because time-series
    pct = weekly.pct_change()

    # Combine into one dataframe
    weekly = weekly.join(pct, rsuffix="_pctchange")

    # Add column indicating number of weeks post-COVID-declaration
    weekly["weeks_after"] = range(len(weekly))

    # Run correlation (rows with NaNs are automatically removed)
    stat = pg.corr(
        weekly["news_pctchange"], weekly["nextDreams_pctchange"], method="spearman"
    )

    # Add number of samples for each, for reporting
    n_dreams, n_news = df.groupby("subreddit").size().loc[["Dreams", "news"]]
    stat["n_rdreams"] = n_dreams
    stat["n_rnews"] = n_news

    stat["r"] = stat["r"].round(4)
    ci = stat.pop("CI95")
    stat.insert(2, "r_upper", ci.str[1])
    stat.insert(2, "r_lower", ci.str[0])
    stat["p_val"] = stat["p_val"].round(5)
    stat["power"] = stat["power"].round(2)
    stat = stat.rename(columns={"p_val": "pval"})

    # Export stats
    stat.to_csv(export_path, index_label="test", sep="\t", encoding="utf-8")
    
    # weekly.to_csv(
    #     export_path_vals, index_label="week", sep="\t", na_rep="N/A", date_format="%Y-%m-%d"
    # )
    return stat, weekly

############################################
################  Plotting  ################
############################################

def plot_correlation(stat, data):
    export_path = export_parent / "correlation-plot.png"

    # Set global matplotlib settings
    utils.load_matplotlib_settings()

    # Select colormap for scatterplot
    colormap = cc.cm.CET_CBTL3_r
    colornorm = plt.Normalize(vmin=1, vmax=data.dropna()["weeks_after"].max())

    # Open figure
    fig, ax = plt.subplots(figsize=(2, 2))

    # Draw data
    ax = sns.scatterplot(
        data=data,
        x="news_pctchange",
        y="nextDreams_pctchange",
        hue="weeks_after",
        palette=colormap,
        hue_norm=colornorm,
        legend=False,
        zorder=10,
        ax=ax,
    )

    # Draw correlation line
    ax = sns.regplot(
        data=data,
        x="news_pctchange",
        y="nextDreams_pctchange",
        color="black",
        scatter=False,
        ci=95,
        n_boot=2000,
        seed=1,
        ax=ax,
    )

    # Draw stats results
    rval, pval = stat.loc["spearman", ["r", "pval"]]
    asterisks = "*" * sum(pval < cutoff for cutoff in [0.05, 0.01, 0.001])
    stats_txt = asterisks + rf"$r$ = {rval:.2f}".replace("0.", ".")
    text_x = 0.6 if year == 2019 else 0.07
    ax.text(text_x, 0.93, stats_txt, ha="left", va="top", transform=ax.transAxes)

    # Adjust aesthetics
    ax.set_xlabel(r"COVID-19 news frequency ${\Delta}_{\%}$")
    ax.set_ylabel(r"Next-week anxious dreaming ${\Delta}_{\%}$")
    xmin, xmax = -0.35, 0.4
    # Bizarre situation where there is an outlier week in 2019 that coincidentally has tones of covid words in it.
    if year == 2019:
        xmin, xmax = -0.75, 1.9
    ylim = 0.6
    if flair == "nondreams" or year == 2019:
        ylim += 0.2
    assert not data["news_pctchange"].le(xmin).any()
    assert not data["news_pctchange"].ge(xmax).any()
    ax.set_xlim(xmin, xmax)
    assert not data["nextDreams_pctchange"].abs().ge(ylim).any()
    ax.set_ylim(-ylim, ylim)
    if year == 2019:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Draw colorbar
    cax = fig.add_axes([0.65, 0.25, 0.2, 0.03])
    smap = plt.cm.ScalarMappable(cmap=colormap, norm=colornorm)
    cbar_ticks = [colornorm.vmin, colornorm.vmax]
    cbar_ticklabels = [str(int(x)) for x in cbar_ticks]
    cbar = fig.colorbar(
        smap, cax=cax, orientation="horizontal", ticklocation="top", ticks=[]
    )
    cbar.outline.set_linewidth(0.5)
    cax.text(
        -0.05, 0.5, cbar_ticklabels[0], ha="right", va="center", transform=cax.transAxes
    )
    cax.text(1.05, 0.5, cbar_ticklabels[1], ha="left", va="center", transform=cax.transAxes)
    cbar_label = "Weeks after\ndeclaration"
    if year == 2019:
        cbar_label = cbar_label.replace("declaration", "March 11, 2019")
    cbar.set_label(cbar_label)

    # Export plots
    utils.save_and_close_fig(export_path, include_svg=True)
    return

#######################################################################################
################  Stats and Plotting for Autocorrelation/Stationarity  ################
#######################################################################################
# Test 4 time-series for autocorrelation and stationarity,
# both subreddits (news and Dreams) and both stages of processing (raw and percent change).

# def autocorrelation_pre():
#     export_path_acor_before = export_parent / "correlation-acor_before.png"

#     # Open up figure
#     fig, axes = plt.subplots(
#         2, 2, figsize=(6, 6), constrained_layout=True, sharex=True, sharey=True
#     )
#     # Select universal plotting keyword arguments
#     acf_kwargs = dict(
#         alpha=0.05,
#         zero=True,
#         missing="drop",
#         title=None,
#         bartlett_confint=False,
#         clip_on=False,
#     )

#     for col, subreddit in enumerate(["news", "Dreams"]):
#         for row, stage in enumerate(["raw", "pctchange"]):
#             ax = axes[row, col]
#             column = f"{subreddit}_{stage}" if stage == "pctchange" else subreddit
#             data = weekly[column].dropna().to_numpy()
#             title = f"COVID-19 on r/{subreddit}, {stage}"
#             if stage == "pctchange":
#                 title = title.replace(stage, r"${\Delta}_{\%}$")

#             # Ljung-Box Q-test for autocorrelation
#             nlags_lb = 10
#             lb = sm.stats.acorr_ljungbox(data, lags=nlags_lb)
#             lb_stat = lb.at[nlags_lb, "lb_stat"]
#             lb_p = lb.at[nlags_lb, "lb_pvalue"]
#             # Compile all stats into text to write on the plots
#             strings = [
#                 f"Ljung-Box (lag={nlags_lb}) = {lb_stat:.1f}, p = {lb_p:.3f}",
#             ]
#             strings = [
#                 s.replace("p = 0.", "p = .").replace("p = .000", "p < .001")
#                 for s in strings
#             ]
#             text = "\n".join(strings)
#             text_pass = "\n".join(
#                 [
#                     "PASS" if lb_p > 0.05 else "FAIL",
#                 ]
#             )
#             # Draw an ACF plot/correlogram to visually inspect autocorrelation
#             sm.graphics.tsa.plot_acf(data, ax, **acf_kwargs)
#             # Draw text
#             ax.text(
#                 0.5,
#                 0.95,
#                 title,
#                 ha="center",
#                 va="top",
#                 weight="bold",
#                 transform=ax.transAxes,
#             )
#             ax.text(0.83, 0.05, text, ha="right", va="bottom", transform=ax.transAxes)
#             ax.text(0.85, 0.05, text_pass, ha="left", va="bottom", transform=ax.transAxes)

#     # Export plots
#     utils.save_and_close_fig(export_path_plot)

#########################################################

def autocorrelation(data):
    export_path_stats = export_parent / "correlation-acor.tsv"
    # Run regression
    # Run correlation (rows with NaNs are automatically removed)
    weekly_nonan = data.dropna(how="any", axis="rows")
    model = sm.formula.ols(
        formula="nextDreams_pctchange ~ news_pctchange", data=weekly_nonan
    )
    result = model.fit()
    
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
    stat, data = run_correlation(flair)
    plot_correlation(stat, data)
    autocorrelation(data)