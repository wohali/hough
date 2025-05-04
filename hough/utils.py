# -*- coding: utf-8 -*-
# vim: ts=4:sw=4:et:tw=88:nowrap

import csv
import datetime
import functools
import logging
import logging.config
import logging.handlers
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from functools import cached_property, reduce
from multiprocessing import Manager, Process, cpu_count, get_context
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Annotated, List, Optional

from cyclopts import App, Parameter, validators
from loguru import logger


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
        _setup_loguru(common.loglevel, common.outpath)
        logger.info(f"=== Run started @ {datetime.datetime.utcnow().isoformat()} ===")
        value = func(*args, **kwargs)
        logger.info(f"=== Run ended  @ {datetime.datetime.utcnow().isoformat()} ===")
        return value

    return wrapper_logit


def check_files(files: list[Path]):
    for f in files:
        if f.exists() and f.is_file():
            continue
        raise
        return f"Cannot access {f}, aborting."
    return None


def abort(pool=None, log_queue=None, listener=None):
    try:
        if pool and isinstance(pool, Pool):
            pool.close()
            pool.terminate()
            pool.join()
            # this lets the producers drain their log queues
            time.sleep(0.1)
            print(
                f"=== Run killed @ {datetime.datetime.utcnow().isoformat()} ===",
                file=sys.stderr,
            )
        if log_queue and listener:
            try:
                log_queue.put(None)
                listener.join()
            except Exception:
                pass
    except Exception:
        print("Exception during abort:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def load_csv(f: Path):
    data = []
    if os.path.exists(f) and os.path.getsize(f) > 0:
        with open(f, newline="") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                if row[0] == "Input File":
                    continue
                for idx in [1, 4, 5]:
                    row[idx] = int(row[idx]) if row[idx] else ""
                data.append([tuple(row)])
    return data


@Parameter(
    name="*", negative=()
)  # flattens namespace so --analyse.debug becomes --debug
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
                raise ValueError(f"Unable to create output directory f{p}")
        return p

    @cached_property
    def debugpath(self) -> Path:
        p = Path(self.outpath, "debug")
        if not p.is_dir():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, FileNotFoundError):
                raise ValueError(f"Unable to create output directory f{p}")
        return p

    @cached_property
    def now(self) -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
