# -*- coding: utf-8 -*-
# vim: ts=4:sw=4:et:tw=88:nowrap

from collections.abc import Generator
from pathlib import Path

import filetype
import imageio.v3 as iio
import pymupdf
from loguru import logger
from numpy import ndarray


def get_pages(
    f: Path,
) -> list[tuple[Path, str, int]]:  # pragma: no cover because dead code
    """Returns the pages in the file, as a list of tuples of the form:
    [(filename, "mime/type", pagenum), ... ]
    """
    try:
        kind = filetype.guess(f)
    except (FileNotFoundError, IsADirectoryError):
        return []
    if kind.mime == "application/pdf":
        pdf = pymupdf.open(f)
        return [(f, kind.mime, x) for x in range(len(pdf))]
    # TODO: Add support for multi-page TIFFs here via
    #   https://imageio.readthedocs.io/en/stable/reference/userapi.html#reading-images
    #   https://imageio.readthedocs.io/en/stable/userapi.html#imageio.mimread
    else:
        # assume single-image file
        return [(f, kind.mime, None)]


def get_num_images(files: list[Path]) -> int:
    ctr = 0

    for f in files:
        kind = filetype.guess(f)
        if not kind:
            return 0
        if kind.mime == "application/pdf":
            pdf = pymupdf.open(f)
            for page in pdf.pages():
                for image in page.get_images():
                    ctr += 1
        else:
            ctr += 1
    return ctr


def get_images(
    files: list[Path],
) -> Generator[
    tuple[
        Path,  # file
        tuple[int, int] | None,  # (pagenum, xref) if multi-image file
        tuple[int, int] | None,  # (xres, yres) dpi
        ndarray,  # image
    ]
]:
    """Generator to return an ndarray for each image in the list of files"""
    for f in files:
        # do not catch exceptions here, let them bubble up
        kind = filetype.guess(f)
        if kind and kind.mime == "application/pdf":
            logger.info(f"Reading PDF {f}...")
            pdf = pymupdf.open(f)
            for page in pdf.pages():
                for image in page.get_images():
                    if image[1] != 0:
                        logger.warning(
                            f"Skipping {f} page {page.number}, non-zero smask {image[1]}"
                        )
                    d = pdf.extract_image(image[0])
                    logger.info(
                        f"Reading PDF {f} page {page.number} image {image[0]}..."
                    )
                    # TODO: add colorspace info?
                    yield (
                        f,
                        (page.number + 1, image[0]),
                        (d["xres"], d["yres"]),
                        iio.imread(d["image"]),
                    )
        # TODO: add support for multi-page TIFFs here
        else:
            try:
                logger.info(f"Reading {f}...")
                yield (f, None, None, iio.imread(f))
            except OSError:
                raise NotImplementedError(f"Unknown file type: {f}")
