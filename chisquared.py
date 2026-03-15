"""
Compare the frequency of nightmares posted on r/Dreams
before and during COVID.

Inputs 1 file:
    - r/Dreams LIWC output, LIWC22 dictionary run on titles

Exports 4 files:
    - chi2 data descriptives as a tsv file
    - chi2 stats as a tsv file
    - chi2 plot as a png file
    - chi2 plot as a pdf file
"""

import argparse

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

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

# Declare filepaths for importing and exporting
derivatives_dir = utils.config["derivatives_directory"]
export_parent = derivatives_dir
if year == 2019:
    export_parent = export_parent / "prioryear"
if flair == "nondreams":
    export_parent = export_parent / "nondreams"
export_parent.mkdir(exist_ok=True)

# Creates pandas datetimes for start, end, COVID declaration
PRE_WINDOW_DURATION = "29D"
POST_WINDOW_DURATION = "30D"
event_date = f"{year}-03-11"
event_dt = pd.to_datetime(event_date, utc=False)
start_dt = event_dt - pd.Timedelta(PRE_WINDOW_DURATION)
end_dt = event_dt + pd.Timedelta(POST_WINDOW_DURATION)
start_date = start_dt.date().isoformat()
end_date = end_dt.date().isoformat()


def run_chisquared(flair):
    export_path = export_parent / "chisquared-stat.tsv"
    # Load data
    df = utils.read_liwc_csv(subreddit="dreams", dream_filter=flair)

    # Reduce to the relevant time period and label pre/post-COVID
    df.index = df.index - pd.Timedelta("1D")  # Shift to account for morning reporting
    df = df.sort_index().loc[start_date:end_date]
    df.loc[:, "PostCovid"] = True
    df.loc[start_date:event_date, "PostCovid"] = False
    # df = df.loc[df["timestamp"].between(start_dt, end_dt, inclusive="both"), :]
    # df["PostCovid"] = df["timestamp"].between(covid_dt, end_dt, inclusive="both")

    # Run stats
    exp, obs, stat = pg.chi2_independence(
        data=df,
        x="PostCovid",
        y="nightmare",
        correction=False,
    )

    # Add observed percentage of nightmares and associated confidence intervals
    obs = obs.rename(columns={0: "dream", 1: "nightmare"})
    obs.index = obs.index.map({False: "PreCovid", True: "PostCovid"}).rename("Covid")
    obs["total"] = obs.sum(axis=1)
    obs["nm_pct"] = obs["nightmare"].div(obs["total"]).mul(100)
    pre_nm_vals = df.query("PostCovid==False")["nightmare"].to_numpy()
    post_nm_vals = df.query("PostCovid==True")["nightmare"].to_numpy()
    bootci_kwargs = dict(func="mean", method="per", n_boot=10000, decimals=6)
    pre_nm_ci = 100 * pg.compute_bootci(pre_nm_vals, **bootci_kwargs)
    post_nm_ci = 100 * pg.compute_bootci(post_nm_vals, **bootci_kwargs)
    obs = obs.sort_index(ascending=False)
    obs["nm_ci_lo"] = [pre_nm_ci[0], post_nm_ci[0]]
    obs["nm_ci_hi"] = [pre_nm_ci[1], post_nm_ci[1]]

    # Add n to stats dataframe for easy access
    stat = stat.set_index("test")
    stat["n"] = obs["total"].sum()
    stat["n_pre"] = obs.at["PreCovid", "total"]
    stat["n_post"] = obs.at["PostCovid", "total"]
    stat["pct_nm_pre"] = obs.at["PreCovid", "nm_pct"]
    stat["pct_nm_post"] = obs.at["PostCovid", "nm_pct"]
    stat["pct_nm_pre_lower"] = pre_nm_ci[0]
    stat["pct_nm_pre_upper"] = pre_nm_ci[1]
    stat["pct_nm_post_lower"] = post_nm_ci[0]
    stat["pct_nm_post_upper"] = post_nm_ci[1]

    stat["lambda"] = stat["lambda"].round(2)
    stat["chi2"] = stat["chi2"].round(3)
    stat["dof"] = stat["dof"].astype(int)
    stat["pval"] = stat["pval"].round(5)
    stat["cramer"] = stat["cramer"].round(2)
    stat["power"] = stat["power"].round(2)

    for col in stat:
        if col.startswith("pct_"):
            stat[col] = stat[col].round(2)
    # # Combine expected and observed frequencies into one dataframe
    # desc = exp.join(obs, lsuffix="_exp", rsuffix="_obs")

    # Export stats
    stat.to_csv(export_path, sep="\t", encoding="utf-8")
    return stat


############################################
################  Plotting  ################
############################################


def plot_chisquared(stat):
    export_path = export_parent / "chisquared-plot.png"
    # Set global matplotlib settings
    utils.load_matplotlib_settings()

    # Select colors
    colormap = cc.cm.cwr
    pre_color = colormap(1.0)
    post_color = colormap(0.0)
    colors = [pre_color, post_color]

    xvals = [0, 1]
    yvals = stat.loc["pearson", ["pct_nm_pre", "pct_nm_post"]].to_numpy()
    ci_lower_cols = [col for col in stat if col.endswith("_lower")]
    ci_upper_cols = [col for col in stat if col.endswith("_upper")]
    ci_lower = stat.loc["pearson", ci_lower_cols].rename("lower").reset_index(drop=True)
    ci_upper = stat.loc["pearson", ci_upper_cols].rename("lower").reset_index(drop=True)
    ci = pd.concat([ci_lower, ci_upper], axis=1).to_numpy()
    evals = abs(yvals - ci.T)

    # Open figure
    fig, ax = plt.subplots(figsize=(1.5, 2))

    # Draw data
    bar_kwargs = dict(width=0.6, linewidth=1, edgecolor="black", error_kw={"lw": 1})
    bars = ax.bar(xvals, yvals, yerr=evals, color=colors, **bar_kwargs)

    # Draw stats results
    chi2val, pval = stat.loc["pearson", ["chi2", "pval"]]
    asterisks = "*" * sum(pval < cutoff for cutoff in [0.05, 0.01, 0.001])
    stats_txt = asterisks + rf"$\chi^2$ = {chi2val:.1f}"
    ax.text(0.5, 0.89, stats_txt, ha="left", va="bottom", transform=ax.transAxes)
    hline_kwargs = dict(lw=1, color="k", capstyle="round")
    ax.hlines(
        y=0.88,
        xmin=xvals[0],
        xmax=xvals[1],
        transform=ax.get_xaxis_transform(),
        **hline_kwargs,
    )

    # Adjust aesthetics
    xtick_labels = ["Before declaration", "After declaration"]
    if year == 2019:
        xtick_labels = [x.replace("declaration", "March 11") for x in xtick_labels]
    _, _, elines = bars.errorbar
    plt.setp(elines, capstyle="round")
    # ax.margins(x=0.2)
    ax.set_ylim(0, 11)
    ax.set_xticks(xvals)
    # ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Nightmare frequency (%)")
    ax.tick_params(
        which="both", top=False, right=False, bottom=False, labelbottom=False
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    bar_hw = bar_kwargs["width"] / 2
    ax.text(0 - bar_hw, 0.2, xtick_labels[0], rotation=90, ha="right", va="bottom")
    ax.text(1 - bar_hw, 0.2, xtick_labels[1], rotation=90, ha="right", va="bottom")
    ax.set_xlim(-0.8, 1.5)

    # Export plots
    utils.save_and_close_fig(export_path, include_svg=True)
    return


if __name__ == "__main__":
    stat = run_chisquared(flair)
    plot_chisquared(stat)
