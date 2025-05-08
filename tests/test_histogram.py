import pytest

from hough import (
    main,
)


def test_just_histogram():
    with pytest.raises(SystemExit) as e:
        main(["histogram", "--results", "newman-results.csv"])
    assert e.value.code == 0


def test_blank_histogram():
    with pytest.raises(SystemExit) as e:
        main(["histogram", "--results", "no-angles.csv"])
    assert e.value.code == 0


def test_invalid_csv_histogram():
    with pytest.raises(SystemExit) as e:
        main(["histogram", "--results", "../README.md"])
    assert e.value.code == 0


def test_null_histogram():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--histogram", "white.jpg"])
    assert e.value.code == 0


def test_histogram_verbose():
    with pytest.raises(SystemExit) as e:
        main(["analyse", "--histogram", "--verbose", "av-000.tif"])
    assert e.value.code == 0


def test_process_histogram_exception(mocker):
    mocker.patch("hough.stats._do_histogram", side_effect=RuntimeError("Boom"))
    with pytest.raises(SystemExit) as e:
        main(["process", "--histogram", "av-000.tif"])
    assert e.value.code == 1


# note: going to get DeprecationWarnings, cannot resolve yet
#   see https://github.com/swig/swig/issues/2881#issuecomment-2332652634
