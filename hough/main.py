#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ts=4:sw=4:et:tw=88:nowrap

#    This file is part of hough, an image deskewing tool
#    Copyright © 2016-2020 Toby Thain <toby@telegraphics.com.au>
#    Copyright © 2020-2025 Joan Touzet <joant@atypical.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import sys
import traceback
from multiprocessing import set_start_method
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, validators
from loguru import logger

from hough import (
    CommonArgs,
    analyse_files,
    check_files,
    histogram as histo,
    logit,
    rotate_files,
)


app = App(default_parameter=Parameter(negative=()))


@app.command()
@logit
def process(
    files: list[Path],
    common: CommonArgs | None = None,
    histogram: Annotated[
        bool, Parameter(help="Display result summary as histogram")
    ] = False,
) -> int:
    """Fully analyse and rotate one or more files."""
    resultspath = analyse_files(files, common)
    if histogram:
        if ret := histo(resultspath) != 0:
            return ret
    return rotate_files(resultspath, common)


@app.command()
@logit
def analyse(
    files: list[Path],
    common: CommonArgs | None = None,
    histogram: Annotated[
        bool, Parameter(help="Display result summary as histogram")
    ] = False,
) -> int:
    """Analyse one or more files for deskewing."""
    if err := check_files(files):
        logger.error(err)
        return 1
    resultspath = analyse_files(files, common)
    if not resultspath:
        return 1
    if histogram:
        return histo(resultspath)
    return 0


@app.command()
# no @logit
def histogram(
    results: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
) -> int:
    """Show a histogram of rotation angles from a previous analysis."""
    return histo(results)


@app.command()
@logit
def rotate(
    results: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True),
            help="Use the specified file for analysis results.",
        ),
    ],
    common: CommonArgs | None = None,
):
    """Rotate one or more files that have previously been analysed."""
    if results.exists():
        results = results.absolute()
    elif str(results) == "results.csv":
        results = Path(common.outpath, "results.csv").absolute()
    elif "TIMESTAMP" in results:
        # support replacing TIMESTAMP if specified
        results = Path(str(results).replace("TIMESTAMP", common.now)).absolute()
    if err := check_files([results]):
        logger.error(err)
        return 1

    # TODO: check all files mentioned in results?
    # if err := check_files(files):
    #    logger.error(err)
    #    return 1
    return rotate_files(results, common)


def main(*args, **kwargs):
    # needed for CUDA backend multiprocessing pool, forced for pytest
    set_start_method("spawn", force=True)
    # TODO: maybe clean up empty out paths on exit?
    try:
        sys.exit(app(*args, **kwargs))
    except Exception:
        print("Uncaught exception:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
