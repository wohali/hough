"""
Worker functions for a parallelizable skew analyser.
"""
import logging
import os
import signal
import traceback

import filetype
import fitz
import numpy as np
import skimage.filters
from imageio import imread, imwrite
from skimage.color import rgb2gray
from skimage.draw import line_aa
from skimage.exposure import is_low_contrast
from skimage.feature import canny
from skimage.morphology import binary_dilation
from skimage.transform import probabilistic_hough_line, rescale, downscale_local_mean
from skimage.util import crop, img_as_ubyte, img_as_uint, invert

import hough

from . import log_utils


# numpy's little helpers


def grey(x):
    return 0.3 if x else 0.0


def bool_to_255(x): # pragma: no cover
    return 255 if x else 0 


def sum(a):
    return a.sum()


bool_to_255f = np.vectorize(bool_to_255)
greyf = np.vectorize(grey)
hough_prec = np.deg2rad(0.02)
hough_theta_h = np.arange(np.deg2rad(-93.0), np.deg2rad(-87.0), hough_prec)
hough_theta_v = np.arange(np.deg2rad(-3.0), np.deg2rad(3.0), hough_prec)
hough_theta_hv = np.concatenate((hough_theta_v, hough_theta_h))


def hough_angles(pos, neg, orientation="H", thresh=(None, None)):

    height, width = pos.shape
    if orientation == "H":
        axis = 1
        theta = hough_theta_h
        fp = np.ones((51, 1))
        margin = int(width * 0.975)
    elif orientation == "V":
        axis = 0
        theta = hough_theta_v
        fp = np.ones((1, 51))
        margin = int(height * 0.025)
    else:
        raise RuntimeError(f"Unknown orientation {orientation}!")
    length = int(height * 0.25)
    blur = skimage.filters.median(neg, footprint=fp, mode="reflect")
    sums = np.apply_along_axis(sum, axis, blur)
    line = sums.argmax(0)
    wsz = max(hough.WINDOW_SIZE, margin)

    # Grab a +/- WINDOW-SIZE strip for evaluation. We've already cropped out the margins.
    if orientation == "H":
        cropped = pos[
            max(line - wsz, 0) : min(
                line + wsz, height
            )  # noqa: E203
        ]
    else:
        cropped = pos[
            :,
            max(line - wsz, 0) : min(
                line + wsz, width
            ),  # noqa: E203
        ]
    edges = binary_dilation(canny(cropped, sigma=2.0, mode="reflect", low_threshold=thresh[0], high_threshold=thresh[1]))
    lines = probabilistic_hough_line(edges, line_length=length, line_gap=2, theta=theta)

    angles = []

    for ((x0, y0), (x1, y1)) in lines:
        # Ensure line is moving rightwards/upwards
        if orientation == "H":
            k = 1 if x1 > x0 else -1
            offset = 0
            horiz = True
        else:
            k = 1 if y1 > y0 else -1
            offset = 90
            horiz = False
        angles.append( (orientation, offset - np.rad2deg(np.math.atan2(k * (y1 - y0), k * (x1 - x0))), x0, y0, x1, y1))

    return (angles, edges)


def _init_worker(queue, debug_arg, now_arg): # pragma: no cover
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    log_utils.setup_queue_logging(queue)
    # this is a global only within the multiprocessing Pool workers, not in the main process.
    global debug, now
    debug = debug_arg
    now = now_arg


def get_pages(f):
    """Returns the pages in the file, as a list of tuples of the form:
        [(filename, "mime/type", pagenum), ... ]
    """
    try:
        kind = filetype.guess(f)
    except (FileNotFoundError, IsADirectoryError):
        return []
    if not kind:
        return [(f, None, None)]
    if kind.mime == "application/pdf":
        pdf = fitz.open(f)
        return [(f, kind.mime, x) for x in range(len(pdf))]
    # TODO: Add support for multi-page TIFFs here via
    #   https://imageio.readthedocs.io/en/stable/userapi.html#imageio.mimread
    else:
        return [(f, kind.mime, None)]


def analyse_page(tuple):
    f, mimetype, page = tuple
    logger = logging.getLogger("hough")
    if page is None:
        # not a known multi-image file
        try:
            image = imread(f)
            logger.info(f"Processing {f}...")
            return [analyse_image(f, image, logger)]
        except ValueError as e:
            # Fall through; might be multi-page anyway...
            logger.debug(f"Single-page read of {f} failed: {e}")
            if debug:
                print(traceback.format_exc())
    if mimetype == "application/pdf":
        results = []
        doc = fitz.open(f)
        imagelist = doc.get_page_images(page)
        for item in imagelist:
            xref = item[0]
            smask = item[1]
            if smask == 0:
                imgdict = doc.extract_image(xref)
                pagenum = float(f"{page + 1}.{xref}")
                logger.info(f"Processing {f} - page {pagenum}...")
                try:
                    image = imread(imgdict["image"])
                    results.append(analyse_image(f, image, logger, pagenum=pagenum))
                except ValueError as e:
                    logger.error(f"Skipping {f} - page {pagenum}: {e}")
            else:
                logger.error(
                    f"Skipping process {f} - page {pagenum} (smask=={smask})"
                )
        return results
    else:
        # TODO: support multi-image TIFF with
        #   https://imageio.readthedocs.io/en/stable/userapi.html#imageio.mimread
        logger.error(f"Cannot process {f}: unknown file format")
        return []


def analyse_file(f):
    results = []
    for page in get_pages(f):
        results += analyse_page(page)
    return results


def analyse_image(f, page, logger, pagenum=None):
    global debug, now
    if "debug" not in globals():
        debug = False
    if "now" not in globals():
        import datetime

        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    pagenum = float(pagenum) if pagenum is not None else ""

    filename = os.path.basename(f)

    if page.ndim > 2:
        logger.debug(f"{f} p{pagenum} is multichannel - converting to greyscale")
        page = rgb2gray(page)
    pageh, pagew = page.shape
    logger.debug(f"Processing {filename} p{pagenum} ({pagew} x {pageh})...")

    # Remove the margins, which are often dirty due to skewing
    # 0.33" of an 8.5" page is approximately 1/25th
    pos = crop(page, pagew // 25)

    if is_low_contrast(pos):
        logger.debug(f"{filename} p{pagenum} - low contrast - blank page?")
        return (f, pagenum, "", "", pagew, pageh)

    # TODO: tweak handle for 0.8 magic value below
    thr = 255 * 0.8
    p = np.clip(pos, 0, thr)
    neg = thr - p
    if debug:
        imwrite(
            f"debug/{now}/{filename}_{pagenum}_neg.png",
            neg,
        )

    h_angles, h_edges = hough_angles(pos, neg, "H")
    v_angles, v_edges = hough_angles(pos, neg, "V")

    if debug:
        imwrite(f"debug/{now}/{filename}_{pagenum}_simple_dilation_h_edges.png",
            img_as_ubyte(h_edges),
        )
        imwrite(f"debug/{now}/{filename}_{pagenum}_simple_dilation_v_edges.png",
            img_as_ubyte(v_edges),
        )

    angles = h_angles + v_angles

    if len(angles) == 0:
        # TODO: more verbose
        if debug:
            imwrite(
                f"debug/{now}/{filename}_{pagenum}_no_hlines.png",
                img_as_ubyte(greyf(h_edges)),
            )
            imwrite(
                f"debug/{now}/{filename}_{pagenum}_no_vlines.png",
                img_as_ubyte(greyf(v_edges)),
            )
        h_angles, h_edges = hough_angles(pos, neg, "H", thresh=(100, 255))
        v_angles, v_edges = hough_angles(pos, neg, "V", thresh=(100, 255))

    if len(angles) > 0:
        angle = np.median([x[1] for x in angles])
        logger.debug(f"{filename} p{pagenum} Hough simple angle: {angle} deg (median)")
        if debug:
            hs = 0
            vs = 0
            h_edges_grey = greyf(h_edges)
            v_edges_grey = greyf(v_edges)
            for result in angles:
                orn, _, x0, y0, x1, y1 = result
                rr, cc, val = line_aa(c0=x0, r0=y0, c1=x1, r1=y1)
                if orn == "H":
                    hs += 1
                    for k, v in enumerate(val):
                        h_edges_grey[rr[k], cc[k]] = (1-v) * h_edges_grey[rr[k], cc[k]] + v
                else:
                    vs += 1
                    for k, v in enumerate(val):
                        v_edges_grey[rr[k], cc[k]] = (1-v) * v_edges_grey[rr[k], cc[k]] + v
            if hs > 0:
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_hlines.png",
                    img_as_ubyte(h_edges_grey),
                )
            if vs > 0:
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_vlines.png",
                    img_as_ubyte(v_edges_grey),
                )
        angles = [x[1] for x in angles]

    else:
        if debug:
            logger.debug(f"{filename} p{pagenum} failed peak sum Hough H/V")

        # We didn't find a good feature at the H or V sum peaks.
        # Let's brutally dilate everything and look for a vertical margin!
        height, width = neg.shape
        if height * width > int(1E6):
            print("shrinking")
            small = downscale_local_mean(neg, (2, 2))
        else:
            small = neg
        t = skimage.filters.threshold_otsu(small)
        dilated = binary_dilation(small > t, np.ones((60,60)))

        edges = canny(dilated, 3)
        edges_grey = greyf(edges)
        lines = probabilistic_hough_line(
            edges, line_length=int(pageh * 0.04), line_gap=6, theta=hough_theta_hv,
        )

        for ((x_0, y_0), (x_1, y_1)) in lines:
            if abs(x_1 - x_0) > abs(y_1 - y_0):
                # angle is <= π/4 from horizontal or vertical
                _, x0, y0, x1, y1 = "H", x_0, y_0, x_1, y_1
            else:
                _, x0, y0, x1, y1 = "V", y_0, -x_0, y_1, -x_1
            # flip angle so that X delta is positive (East quadrants).
            k = 1 if x1 > x0 else -1
            a = np.rad2deg(np.math.atan2(k * (y1 - y0), k * (x1 - x0)))

            # Zero angles are suspicious -- could be a cropping margin.
            # If not, they don't add information anyway.
            if a != 0:
                angles.append(-a)
                rr, cc, val = line_aa(c0=x_0, r0=y_0, c1=x_1, r1=y_1)
                for k, v in enumerate(val):
                    edges_grey[rr[k], cc[k]] = (1 - v) * edges_grey[rr[k], cc[k]] + v

        if angles:
            angle = np.median(angles)
            logger.debug(
                f"{filename} p{pagenum} dilated Hough angle: {angle} deg (median)"
            )
            if debug:
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilated.png",
                    bool_to_255f(img_as_ubyte(dilated)),
                )
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_{angle}_lines.png",
                    img_as_ubyte(edges_grey),
                )
                #imwrite(
                #    f"debug/{now}/{filename}_{pagenum}_{angle}_lines_verticaldilated.png",
                #    bool_to_255f(img_as_ubyte(dilated)),
                #)
        else:
            angle = None
            logger.debug(f"{filename} p{pagenum} failed dilated Hough V")
            if debug:
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilated.png",
                    bool_to_255f(img_as_ubyte(dilated)),
                )
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilate_edges_grey.png",
                    img_as_ubyte(edges_grey),
                )
                imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilate_edges.png",
                    bool_to_255f(img_as_ubyte(edges)),
                )

    return (
        f,
        pagenum,
        angle if angle is not None else "",
        np.var(angles) if angles else "",
        pagew,
        pageh,
    )
