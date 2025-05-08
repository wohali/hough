from pathlib import Path

import pytest

from hough import (
    main,
)


def test_smask_pdf():
    with pytest.raises(SystemExit) as e:
        main(["process", "smask.pdf"])
    assert e.value.code == 0


def test_process_img():
    with pytest.raises(SystemExit) as e:
        main(["process", "deep_hurting.png"])
    assert e.value.code == 0


@pytest.mark.parametrize("outpath", ["t"])
def test_full_process():
    with pytest.raises(SystemExit) as e:
        main(
            [
                "process",
                "--histogram",
                "--out=t",
                "av-000.tif",
                "av-001.jpg",
            ]
        )
    assert e.value.code == 0
    assert Path("t").exists()
    assert Path("t/results.csv").exists()
    assert Path("t/av-000.tif").exists()
    # may not end up rotating
    # assert Path("t/av-001.jpg").exists()


# note: going to get DeprecationWarnings, cannot resolve yet
#   see https://github.com/swig/swig/issues/2881#issuecomment-2332652634
