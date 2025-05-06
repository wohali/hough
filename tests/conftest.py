import os
from pathlib import Path
from shutil import rmtree

import pytest


@pytest.fixture(scope="session")
def clean_sampledir():  # pragma: no cover
    for f in Path("samples").glob(".coverage.*"):
        f.unlink(missing_ok=True)


@pytest.fixture
def sampledir():
    old_cwd = os.getcwd()
    os.chdir("samples")
    rmtree("out", ignore_errors=True)
    yield
    rmtree("out", ignore_errors=True)
    os.chdir(old_cwd)
