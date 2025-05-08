from pathlib import Path

import pytest

from hough import (
    analyse_files,
    main,
    CommonArgs,
    load_csv,
)


def test_unorientable():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "unorientable.jpg"])
    assert e.value.code == 0


def test_unorientable_debug():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--debug", "unorientable.jpg"])
    assert e.value.code == 0


def test_low_contrast():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "black.png"])
    assert e.value.code == 0


def test_white():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "white.jpg"])
    assert e.value.code == 0


def test_no_lines():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "rhombus.jpg"])
    assert e.value.code == 0


def test_small_blob():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "blob.png"])
    assert e.value.code == 0


def test_small_image():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "deep_hurting.png"])
    assert e.value.code == 0


def test_smask_pdf():
    with pytest.raises(SystemExit) as e:
        main(["process", "smask.pdf"])
    assert e.value.code == 0


def test_vert():
    resultspath = analyse_files([Path("av-000.tif")], CommonArgs())
    results = load_csv(resultspath)
    res = results[0][0]
    assert res[0] == "av-000.tif"
    assert res[1] == 0.0
    assert res[2] > -2 and res[2] <= 0.0
    assert res[3] < 1.0
    # assert(res[4] == 464)
    # assert(res[5] == 645)


def test_hv_fail():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "av-001.jpg"])
    assert e.value.code == 0


def test_unstraightenable():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "binder.png"])
    assert e.value.code == 0


@pytest.mark.parametrize("outpath", ["t"])
def test_rgb_pdf_4_workers():
    with pytest.raises(SystemExit) as e:
        main(
            [
                "analyse",
                "--debug",
                "--out=t",
                "--workers=4",
                "Newman_Computer_Exchange_VAX_PC_PDP11_Values.pdf",
            ]
        )
    assert e.value.code == 0
    assert Path("t/results.csv").exists()
    assert Path("t/hough.log").exists()
    assert Path("t/debug").exists()
    assert Path("t/debug").is_dir()


def test_form_pdf():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "i-9.pdf"])
    assert e.value.code == 0


def test_mixed_pdf():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "i-9-paper-version.pdf"])
    assert e.value.code == 0


def test_multipage_compressed_tiff():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "dg-nova-multipage.tif"]) == 0
    assert e.value.code == 0


def test_batch_tiff():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "av-multi-batch.tif"]) == 0
    assert e.value.code == 0


def test_unknown():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "../README.md"])
    assert e.value.code == 1


# note: going to get DeprecationWarnings, cannot resolve yet
#   see https://github.com/swig/swig/issues/2881#issuecomment-2332652634
