# -*- coding: utf-8 -*-
# vim: ts=4:sw=4:et:tw=88:nowrap

import csv
import datetime
import functools
import sys
import time
import traceback
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import get_context, Process, Queue, Manager
from multiprocessing.pool import Pool
from pathlib import Path
from tqdm import tqdm

from cyclopts import Parameter
from loguru import logger

try:  # pragma: no cover
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    import numpy as cp
import numpy as np


def _setup_loguru(log_level, outpath):
    spawn_context = get_context("spawn")
    fmt = "{time:YYYY-MM-DD HH:mm:ss,SSS} {level:8} {process.name:>18}-{process.id:<7} {message}"
    # reattach stderr sink with queueing to ensure inheritability
    logger.remove()
    logger.add(
        sys.stderr,
        context=spawn_context,
        enqueue=True,
        level=log_level,
        format="{message}",
    )
    # TODO: do we still want this?
    logger.add(
        Path(outpath, "hough.log"),
        context=spawn_context,
        enqueue=True,
        level="DEBUG",
        format=fmt,
    )


def logit(func):
    """Initialise a function for logging"""

    @functools.wraps(func)
    def wrapper_logit(*args, **kwargs):
        # there must be an easier way
        if any(isinstance(val, CommonArgs) for val in args):
            for val in args:
                if isinstance(val, CommonArgs):
                    common = val
        else:
            common = CommonArgs()
            args = args + (common,)
        try:
            _setup_loguru(common.loglevel, common.outpath)
        except ValueError:
            print(
                "Unable to create output directory or logfile! Aborting.",
                file=sys.stderr,
            )
            exit(1)

        logger.info(
            f"=== Run started @ {datetime.datetime.now(datetime.UTC).isoformat()} ==="
        )
        value = func(*args, **kwargs)
        logger.info(
            f"=== Run ended  @ {datetime.datetime.now(datetime.UTC).isoformat()} ==="
        )
        return value

    return wrapper_logit


def get_cpu_image(img: cp.ndarray | np.ndarray) -> np.ndarray:  # pragma: no cover
    if "cuda" in dir(cp):
        return img.get()
    else:
        return img


def get_gpu_image_maybe(
    img: np.ndarray, dtype=np.float32
) -> cp.ndarray:  # pragma: no cover
    if "cuda" in dir(cp):
        return cp.asarray(img, dtype)
    else:
        return img


def _pbar_listener(q, total, disable, unit, desc):
    pbar = tqdm(total=total, disable=disable, unit=unit, desc=desc)
    for item in iter(q.get, None):
        pbar.update()


def start_pbar(total: int, disable: bool, unit: str, desc: str) -> (Queue, Process):
    q = Manager().Queue(-1)
    proc = Process(
        target=_pbar_listener,
        args=(q, total, disable, unit, desc),
    )
    proc.start()
    return q, proc


def abort(pool=None, log_queue=None, listener=None):
    try:
        if pool and isinstance(pool, Pool):
            pool.close()
            pool.terminate()
            pool.join()
            # this lets the producers drain their log queues
            time.sleep(0.1)
            print(
                f"=== Run killed @ {datetime.datetime.now(datetime.UTC).isoformat()} ===",
                file=sys.stderr,
            )
        if log_queue and listener:
            try:
                log_queue.put(None)
                listener.join()
            except Exception:
                pass
    except Exception:  # pragma: no cover
        print("Exception during abort:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def load_csv(f: Path):
    data = []
    try:
        with open(f, newline="") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                if row[0] == "Input File":
                    continue
                # for idx in [4, 5, 6, 7]:
                for idx in [4, 5]:
                    row[idx] = int(row[idx]) if row[idx] else ""
                row[2] = row[2] if row[2] else 0
                data.append([tuple(row)])
    except (OSError, ValueError) as e:
        logger.error(f"Unable to read results CSV {f}: {e}")
    return data


def validate_common(type_, value):
    if value and value.debug:
        # this will raise ValueError if there is an issue
        _ = value.debugpath


# name="*" flattens namespace so --analyse.debug becomes --debug
@Parameter(name="*", negative=(), validator=validate_common)
@dataclass
class CommonArgs:
    debug: bool = False
    "Save intermediate results in debug/ under out folder."

    verbose: bool = False
    "Print status messages instead of progress bar."

    out: Path = Path("out/TIMESTAMP")
    "Use the specified path for results and post-rotated files."

    workers: int = 4
    "Number of workers to run simultaneously."

    def __post_init__(self):
        if self.debug:
            self.loglevel = "DEBUG"
        elif self.verbose:
            self.loglevel = "INFO"
        else:
            self.loglevel = "WARNING"
        self.logger = logger

    @cached_property
    def outpath(self) -> Path:
        if "TIMESTAMP" in self.out.parts:
            self.out = Path(str(self.out).replace("TIMESTAMP", self.now))
        p = self.out.absolute()
        if not p.is_dir():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, FileNotFoundError):
                raise ValueError(f"Unable to create output directory {p}")
        return p

    @cached_property
    def debugpath(self) -> Path:
        p = Path(self.outpath, "debug")
        if not p.is_dir():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, FileNotFoundError):
                raise ValueError(f"Unable to create debug directory {p}")
        return p

    @cached_property
    def now(self) -> str:
        return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%M%SZ")
