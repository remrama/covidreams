"""
Initialize the data directory structure used throughout the rest of the scripts.
Directory names are specified in the config.json configuration file.
"""
from pathlib import Path
import utils

config = utils.load_config()

data_directories = [
    config["source_directory"],  # for raw data, no touchey
    config["derivatives_directory"],  # for analysis output
]

for directory in data_directories:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=False)
