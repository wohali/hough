import os
from multiprocessing import Pool, set_start_method
from pathlib import Path
from shutil import rmtree

import pytest

from hough import (
    __version__,
    analyse_files,
    main,
    abort,
    start_pbar,
    CommonArgs,
    load_csv,
)


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    set_start_method("spawn", force=True)


@pytest.fixture(autouse=True)
def print_newline():
    # Makes output cleaner when progress bar is drawn
    print()


@pytest.mark.usefixtures("clean_sampledir")
def test_version():
    assert __version__


@pytest.mark.usefixtures("sampledir")
def test_histogram_exception(mocker):
    mocker.patch("hough.stats._do_histogram", side_effect=RuntimeError("Boom"))
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--histogram", "av-000.tif"])
    assert e.value.code == 1


@pytest.mark.usefixtures("sampledir")
def test_null_histogram():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--histogram", "white.jpg"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_abort(tmpdir):
    q, proc = start_pbar(1, True, "tests", "Testing: ")
    with Pool(processes=2) as p:
        abort(p, q, proc)
    with Pool(processes=2) as p:
        abort(p, 123, "abc")
    with Pool(processes=2) as p:
        abort(p, None, None)
    abort(123, 456, "abc")


@pytest.mark.usefixtures("sampledir")
def test_no_valid_files():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "a", "b", "c"])
    assert e.value.code == 1


@pytest.mark.usefixtures("sampledir")
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


@pytest.mark.usefixtures("sampledir")
def test_hv_fail():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "av-001.jpg"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_multifile_debug():
    rmtree("debug", ignore_errors=True)
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "--debug", "av-000.tif", "av-001.jpg"])
    assert e.value.code == 0
    assert Path("t/debug").exists()
    assert Path("t/hough.log").exists()
    # TODO: assert expected output appears under debug/DATE/whatever


@pytest.mark.usefixtures("sampledir")
def test_unstraightenable():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=t", "--debug", "binder.png"])
    assert e.value.code == 0
    assert Path("t/hough.log").exists()
    os.remove("t/hough.log")
    rmtree("t", ignore_errors=True)


@pytest.mark.usefixtures("sampledir")
def test_unstraightenable_nodebug():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "binder.png"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_just_histogram():
    with pytest.raises(SystemExit) as e:
        main(["histogram", "--results", "newman-results.csv"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_blank_histogram():
    with pytest.raises(SystemExit) as e:
        main(["histogram", "--results", "no-angles.csv"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_histogram_verbose():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--histogram", "--verbose", "av-000.tif"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_rgb_pdf_4_workers():
    if Path("t").exists():
        rmtree("t", ignore_errors=True)
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
    rmtree("t", ignore_errors=True)


@pytest.mark.usefixtures("sampledir")
def test_form_pdf():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "i-9.pdf"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_mixed_pdf():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "i-9-paper-version.pdf"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_white():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "white.jpg"])
    assert e.value.code == 0


@pytest.mark.usefixtures("sampledir")
def test_unknown():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "../README.md"])
    assert e.value.code == 1


@pytest.mark.usefixtures("sampledir")
def test_broken_out():
    open(".__pytest", "a").close()
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--out=.__pytest", "white.jpg"])
    assert e.value.code == 1
    os.remove(".__pytest")


@pytest.mark.usefixtures("sampledir")
def test_no_args():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code is None


@pytest.mark.usefixtures("sampledir")
def test_rotate_nothing():
    if Path(".__pytest").exists():
        os.remove(".__pytest")
    open(".__pytest", "a").close()
    with pytest.raises(SystemExit) as e:
        main(["rotate", "--results=.__pytest"])
    assert e.value.code == 1
    os.remove(".__pytest")


@pytest.mark.usefixtures("sampledir")
def test_rotate():
    if Path(".__pytest.dir").exists():
        rmtree(".__pytest.dir")
    with pytest.raises(SystemExit) as e:
        main(
            [
                "process",
                "--histogram",
                "--out=.__pytest.dir",
                "Newman_Computer_Exchange_VAX_PC_PDP11_Values.pdf",
                "av-000.tif",
            ]
        )
    assert e.value.code == 0
    assert Path(".__pytest.dir").exists()
    assert Path(".__pytest.dir/results.csv").exists()
    assert Path(
        ".__pytest.dir/Newman_Computer_Exchange_VAX_PC_PDP11_Values.pdf"
    ).exists()
    assert Path(".__pytest.dir/av-000.tif").exists()
    rmtree(".__pytest.dir")


@pytest.mark.usefixtures("sampledir")
def test_rotate_from_results():
    if Path(".__pytest.dir").exists():
        rmtree(".__pytest.dir")
    with pytest.raises(SystemExit) as e:
        main(["rotate", "--out=.__pytest.dir", "--results=newman-results.csv"]) == 0
    assert e.value.code == 0
    assert Path(
        ".__pytest.dir/Newman_Computer_Exchange_VAX_PC_PDP11_Values.pdf"
    ).exists()
    rmtree(".__pytest.dir")


# note: going to get DeprecationWarnings, cannot resolve yet
#   see https://github.com/swig/swig/issues/2881#issuecomment-2332652634
