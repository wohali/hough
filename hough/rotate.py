"""
Worker functions for a parallelizable deskewer.
"""

import signal
from multiprocessing import Queue
from multiprocessing.pool import Pool
from pathlib import Path

from loguru import logger

from . import start_pbar, get_cpu_image, get_gpu_image_maybe

try:  # pragma: no cover
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
except ModuleNotFoundError:  # pragma: no cover
    import numpy as cp
    import scipy.ndimage as ndi

import filetype
import fitz
import imageio.v3 as iio
import numpy as np

from .utils import CommonArgs, abort, load_csv


# TODO: provide order override?
def rotate_ndarray(cpuimg: np.ndarray, angle: float, order=0) -> np.ndarray:
    gpuimg = get_gpu_image_maybe(cpuimg, dtype=cp.uint8)
    rotatedimg = ndi.rotate(gpuimg, -angle, mode="nearest", reshape=False, order=order)
    return get_cpu_image(rotatedimg)


def rotate(imagelist: list[tuple]) -> int:
    """Actually rotates a single file of 1+ images.
    Rotated file has the same name and is placed in common.outpath.
    """
    guess = filetype.guess(imagelist[0][0])
    if guess and guess.mime == "application/pdf":
        newdoc = fitz.open()
    else:
        newdoc = None
    filename = Path(imagelist[0][0])
    basename = filename.name
    kind = filetype.guess(filename)
    logger = common.logger

    for image in imagelist:
        page = int(image[1]) if image[1] else ""
        angle = float(image[2]) if image[2] else 0.0
        if angle == 0.0:
            continue
        if not page:
            # single-image file, not a container
            logger.info(f"Rotating {filename}...")
            cpuimg = iio.imread(image[0])
            fixed = rotate_ndarray(cpuimg, angle)
            iio.imwrite(f"{common.outpath}/{basename}", fixed)
            queue.put(1)
        else:
            if kind and kind.mime == "application/pdf":
                doc = fitz.open(image[0])
                imagelist = doc.get_page_images(page - 1)
                # TODO: Correctly deal with multiple images on a page
                for item in imagelist:
                    xref = item[0]
                    smask = item[1]
                    if smask == 0:
                        imgdict = doc.extract_image(xref)
                        logger.info(
                            f"Rotating {filename} - page {page} - xref {xref}..."
                        )
                        try:
                            cpuimg = iio.imread(imgdict["image"])
                            fixed = rotate_ndarray(cpuimg, angle)
                            imgext = imgdict["ext"]
                            imgbytes = iio.imwrite(
                                "<bytes>", fixed, extension=("." + imgext)
                            )
                            imgdoc = fitz.open(stream=imgbytes, filetype=imgext)
                            rect = imgdoc[0].rect
                            pdfbytes = imgdoc.convert_to_pdf()
                            imgdoc.close()
                            img_pdf = fitz.open("pdf", pdfbytes)
                            page = newdoc.new_page(width=rect.width, height=rect.height)
                            page.show_pdf_page(rect, img_pdf, 0)
                        except ValueError as e:
                            logger.error(
                                f"Skipping rotating {filename} - page {page} - xref {xref}: {e}"
                            )
                    else:
                        logger.error(
                            f"Skipping process {filename} - page {page} - image {xref} (smask=={smask})"
                        )
            # TODO: deal with other multi-image formats
            else:
                logger.error(
                    f"Skipping file {filename} - unknown multi-page file format"
                )
        queue.put(1)
    if newdoc:
        logger.info(f"Saving {common.outpath}/{basename}...")
        newdoc.save(f"{common.outpath}/{basename}")


# NOTE: _init_worker must be in the same module as analyse_files (i.e. the call to the initializer), not sure why
def _init_rotator(common_arg: CommonArgs, q: Queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global common, logger, queue
    common = common_arg
    logger = common_arg.logger
    queue = q


def rotate_files(resultsfile: Path, common: CommonArgs) -> int:
    """Rotate all of the files mentioned in the results.csv resultsfile."""
    # TODO: further parallelise this by splitting multi-image files first?

    # load a previously generated results.csv file
    results = load_csv(resultsfile)
    if len(results) == 0:
        logger.error("Nothing to do!")
        return 1

    # create dict of lists, each key is a file and each value a list of tuple-images
    dictresults = {}
    for result in results:
        for image in result:
            dictresults.setdefault(image[0], []).append(image)
    sortedresults = [
        sorted(v, key=lambda x: int(x[1]) if x[1] else 0)
        for k, v in dictresults.items()
    ]

    try:
        q, proc = start_pbar(
            len(results),
            (common.loglevel != "WARNING"),
            "pg",
            "Rotation: ",
        )
        with Pool(
            processes=common.workers,
            initializer=_init_rotator,
            initargs=(common, q),
        ) as p:
            for i, result in enumerate(
                p.imap_unordered(rotate, sortedresults, min(common.workers, 8))
            ):
                pass
            # see https://coverage.readthedocs.io/en/7.8.0/subprocess.html#using-multiprocessing
            p.close()
            p.join()
        q.put(None)
        proc.join()
        return 0

    except KeyboardInterrupt:
        import sys

        print("Caught KeyboardInterrupt, terminating workers...", file=sys.stderr)
        abort(p, q, proc)
        return 1
