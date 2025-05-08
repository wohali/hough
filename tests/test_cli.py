import os
import sys
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import pytest

from hough import (
    __version__,
    main,
    abort,
    start_pbar,
)


def test_version():
    assert __version__


def test_abort():
    q, proc = start_pbar(1, True, "tests", "Testing: ")
    with Pool(processes=2) as p:
        abort(p, q, proc)
    with Pool(processes=2) as p:
        abort(p, 123, "abc")
    with Pool(processes=2) as p:
        abort(p, None, None)
    abort(123, 456, "abc")


def test_no_valid_files():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "a", "b", "c"])
    assert e.value.code == 1


@pytest.mark.parametrize("outpath", ["t"])
def test_multifile_debug(sampledir):
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "--debug", "av-000.tif", "av-001.jpg"])
    assert e.value.code == 0
    assert Path("t/debug").exists()
    assert Path("t/hough.log").exists()
    # TODO: assert expected output appears under debug/DATE/whatever


@pytest.mark.parametrize("outpath", ["t"])
def test_reuse_outpath():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "--debug", "binder.png"])
    assert e.value.code == 0
    assert Path("t/hough.log").exists()
    assert Path("t/results.csv").exists()
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "--debug", "rhombus.jpg"])
    assert e.value.code == 0
    assert Path("t/hough.log").exists()
    assert Path("t/results.csv").exists()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="perm error in CI")
def test_broken_debug():
    Path("t").mkdir()
    open("t/debug", "a").close()
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "--debug", "deep_hurting.png"])
    assert e.value.code == 1
    rmtree("t")


def test_no_args():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code is None


@pytest.mark.skipif(sys.platform.startswith("win"), reason="perm error in CI")
def test_analyse_broken_out():
    open("t", "a").close()
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "white.jpg"])
    assert e.value.code == 1
    os.remove("t")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="perm error in CI")
@pytest.mark.parametrize("outpath", ["t"])
def test_rotate_broken_out():
    open("t", "a").close()
    with pytest.raises(SystemExit) as e:
        main(["rotate", "--results=t"])
    assert e.value.code == 1
    Path("t").unlink()


# note: going to get DeprecationWarnings, cannot resolve yet
#   see https://github.com/swig/swig/issues/2881#issuecomment-2332652634
