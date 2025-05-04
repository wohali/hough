from importlib import metadata

import toml

from .utils import CommonArgs, abort, check_files, load_csv, logit
from .multifiles import get_images, get_num_images
from .stats import histogram
from .analyse import analyse_files, analyse_image
from .rotate import rotate_files


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = toml.load("pyproject.toml")["tool"]["poetry"]["version"] + "dev"
