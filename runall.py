"""Run all analysis scripts in the order"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"Running {cmd} …")
    result = subprocess.run([sys.executable] + cmd.split(), check=True)
    return result


scraping_scripts = [
    "scrape_reddit.py -r Dreams --start 2019-01-01 --end 2020-12-31",
    "scrape_reddit.py -r Dreams --start 2019-01-01 --end 2020-12-31",
    "scrape_reddit.py -r news --start 2019-01-01 --end 2020-12-31",
]

liwc_scripts = [
    "merge_news.py",
    "liwc_run.py",
    "merge_liwc.py",
]

analysis_scripts = [
    "example_view.py",
    "samplesize.py",
    "regression.py",
    "chisquared.py",
    "correlation.py",
    # Prior year controls (seasonality)
    "samplesize.py --prioryear",
    "regression.py --prioryear",
    "chisquared.py --prioryear",
    "correlation.py --prioryear",
    # Non-flaired post controls (daily language)
    "regression.py --nondreams",
    "chisquared.py --nondreams",
    "correlation.py --nondreams",
    # Longer post-COVID window
    "regression.py --longerwindow",
    # Generate manuscript assets
    "generate_figure.py",
    "generate_table.py",
    "generate_variables.py",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all analysis scripts.")
    parser.add_argument(
        "--scrape", action="store_true", help="Include data scraping step"
    )
    parser.add_argument(
        "--liwc", action="store_true", help="Include LIWC processing step"
    )
    args = parser.parse_args()

    scripts = []
    if args.scrape:
        scripts += scraping_scripts
    if args.liwc:
        scripts += liwc_scripts
    scripts += analysis_scripts

    for cmd in scripts:
        run(cmd)
    print("\nAll scripts completed successfully.")
