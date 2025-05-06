import math

import numpy as np
from pathlib import Path
import termplotlib as tpl

from loguru import logger

from .utils import load_csv


def _do_histogram(results: Path):
    # load the file
    data = load_csv(results)
    res = []
    for r in data:
        for image in r:
            res.append(abs(image[2]))
    if not res:
        print("\nNo angles found, so no histogram.")
        return

    counts, bin_edges = np.histogram(res, bins=20, range=(0, int(math.ceil(max(res)))))
    labels = [
        "{:.2f}° - {:.2f}°".format(bin_edges[k], bin_edges[k + 1])
        for k in range(len(bin_edges) - 1)
    ]
    fig = tpl.figure()
    fig.barh(
        counts,
        labels=labels,
        max_width=40,
        bar_width=1,
        show_vals=True,
        force_ascii=False,
    )

    print("\n=== Skew statistics ===")
    fig.show()

    print(f"Samples: {len(res)}")
    print(f"50th percentile: {np.percentile(res, 50):1.2f}°")
    print(f"90th percentile: {np.percentile(res, 90):1.2f}°")
    print(f"99th percentile: {np.percentile(res, 99):1.2f}°")


def histogram(results: Path):
    try:
        _do_histogram(results)
        return 0
    except Exception:
        import traceback

        logger.error(f"Exception in histogram process: \n{traceback.format_exc()}")
        return 1
