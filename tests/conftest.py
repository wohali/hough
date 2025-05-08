import os
from multiprocessing import set_start_method
from pathlib import Path
from shutil import rmtree

import pytest


# should not be necessary, but can't hurt.
@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    set_start_method("spawn", force=True)


# always start a test run with clean coverage data
@pytest.fixture(scope="session", autouse=True)
def clean_sampledir():  # pragma: no cover
    for f in Path("samples").glob(".coverage.*"):
        f.unlink(missing_ok=True)
    yield


# Makes output cleaner when progress bar is drawn
@pytest.fixture(autouse=True)
def print_newline():
    print()
    yield
    print()


@pytest.fixture
def outpath():
    return "out"


# Run tests from samples directory, auto-cleaning up output directory
@pytest.fixture(autouse=True)
def sampledir(outpath):
    old_cwd = os.getcwd()
    os.chdir("samples")
    rmtree(outpath, ignore_errors=True)
    yield
    rmtree(outpath, ignore_errors=True)
    os.chdir(old_cwd)
