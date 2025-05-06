from importlib import metadata

import toml

from .utils import (
    CommonArgs,
    abort,
    load_csv,
    logit,
    start_pbar,
    get_cpu_image,
    get_gpu_image_maybe,
)
from .multifiles import get_images, get_num_images
from .stats import histogram
from .analyse import analyse_files, analyse_image
from .rotate import rotate_files
from .main import main


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = toml.load("pyproject.toml")["tool"]["poetry"]["version"] + "dev"

__all__ = [
    "CommonArgs",
    "abort",
    "load_csv",
    "logit",
    "start_pbar",
    "get_cpu_image",
    "get_gpu_image_maybe",
    "get_images",
    "get_num_images",
    "histogram",
    "analyse_files",
    "analyse_image",
    "rotate_files",
    "main",
]
