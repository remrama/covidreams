"""Run all analysis scripts in the order"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"Running {' '.join(cmd)} …")
    result = subprocess.run([sys.executable] + cmd[1:], check=True)
    return result


scraping_scripts = [
    ["python", "scrape_reddit.py", "-r", "Dreams", "--start", "2019-01-01", "--end", "2020-12-31"],
    ["python", "scrape_reddit.py", "-r", "news", "--start", "2019-01-01", "--end", "2020-12-31"],
]

liwc_scripts = [
    ["python", "merge_news.py"],
    ["python", "liwc_run.py"],
    ["python", "merge_liwc.py"],
]

analysis_scripts = [
    ["python", "example_view.py"],
    ["python", "samplesize.py"],
    ["python", "regression.py"],
    ["python", "chisquared.py"],
    ["python", "correlation.py"],

    # Prior year controls (seasonality)
    ["python", "samplesize.py", "--prioryear"],
    ["python", "regression.py", "--prioryear"],
    ["python", "chisquared.py", "--prioryear"],
    ["python", "correlation.py", "--prioryear"],

    # Non-flaired post controls (daily language)
    ["python", "regression.py", "--nondreams"],
    ["python", "chisquared.py", "--nondreams"],
    ["python", "correlation.py", "--nondreams"],

    # Longer post-COVID window
    ["python", "regression.py", "--longerwindow"],

    # Compile multi-panel figure
    ["python", "figure.py"]
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all analysis scripts.")
    parser.add_argument("--scrape", action="store_true", help="Include data scraping step")
    parser.add_argument("--liwc", action="store_true", help="Include LIWC processing step")
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
