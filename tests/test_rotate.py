import sys
from pathlib import Path

import pytest

from hough import (
    main,
)


def test_rotate_from_results():
    with pytest.raises(SystemExit) as e:
        main(["rotate", "--out=out", "--results=newman-results.csv"]) == 0
    assert e.value.code == 0
    assert Path("out/Newman_Computer_Exchange_VAX_PC_PDP11_Values.pdf").exists()


def test_rotate_smask_pdf():
    with pytest.raises(SystemExit) as e:
        main(["rotate", "--results=smask.csv"])
    assert e.value.code == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="perm error in CI")
@pytest.mark.parametrize("outpath", ["t"])
def test_rotate_broken_out():
    open("t", "a").close()
    with pytest.raises(SystemExit) as e:
        main(["rotate", "--results=t"])
    assert e.value.code == 1
    Path("t").unlink()


def test_rotate_nothing_two():
    with pytest.raises(SystemExit) as e:
        main(["rotate", "nothing.at.all"])
    assert e.value.code == 1


# note: going to get DeprecationWarnings, cannot resolve yet
#   see https://github.com/swig/swig/issues/2881#issuecomment-2332652634
