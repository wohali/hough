"""
Worker functions for a parallelizable skew analyser.
"""
import logging
import math
import os
import signal
import traceback

import cupy as cp
import cupyx.scipy.ndimage as ndi
import cv2
import filetype
import imageio.v3 as iio
import numpy as np
import pymupdf
from cucim.skimage.color import rgb2gray
from cucim.skimage.exposure import is_low_contrast
from cucim.skimage.feature import canny
from cucim.skimage.filters import threshold_otsu
from cucim.skimage.morphology import footprint_rectangle
from cucim.skimage.transform import downscale_local_mean
from cucim.skimage.util import crop, img_as_bool, img_as_ubyte
from skimage.draw import line_aa
from skimage.transform import probabilistic_hough_line

from . import log_utils


# TODO: parametrise out all magic numbers
# TODO: deal with the logger issue, maybe via loguru


# hough's little helpers
def _grey(x):
    return 0.3 if x else 0.0


# vectorize applies JIT compiler to the given function
_greyf = cp.vectorize(_grey)
# CuPy scalars are actually GPU-resident zero-dimensional arrays, so we use numpy here instead
hough_prec = np.deg2rad(0.02)
hough_theta_h = np.arange(
    np.deg2rad(-93.0), np.deg2rad(-87.0), hough_prec, dtype=cp.float64
)
hough_theta_v = np.arange(
    np.deg2rad(-3.0), np.deg2rad(3.0), hough_prec, dtype=cp.float64
)
hough_theta_hv = np.concatenate((hough_theta_v, hough_theta_h))


# thank you, https://github.com/rapidsai/cucim/issues/329 !
def cv_cuda_gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    assert len(arr.shape) in (
        2,
        3,
    ), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"
    type_map = {
        cp.dtype("uint8"): cv2.CV_8U,
        cp.dtype("int8"): cv2.CV_8S,
        cp.dtype("uint16"): cv2.CV_16U,
        cp.dtype("int16"): cv2.CV_16S,
        cp.dtype("int32"): cv2.CV_32S,
        cp.dtype("float32"): cv2.CV_32F,
        cp.dtype("float64"): cv2.CV_64F,
    }
    depth = type_map.get(arr.dtype)
    assert depth is not None, f"Unsupported CuPy array dtype {arr.dtype}"
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
    # (depth&7) + ((channels - 1) << 3)
    mat_type = depth + ((channels - 1) << 3)
    # TODO: do we need [1::-1] here to invert the matrix?
    mat = cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__["shape"][1::-1],
        mat_type,
        arr.__cuda_array_interface__["data"][0],
    )
    return mat


def cp_array_from_cv_cuda_gpumat(mat: cv2.cuda.GpuMat) -> cp.ndarray:
    class CudaArrayInterface:
        def __init__(self, gpu_mat: cv2.cuda.GpuMat):
            w, h = gpu_mat.size()
            type_map = {
                cv2.CV_8U: "|u1",
                cv2.CV_8S: "|i1",
                cv2.CV_16U: "<u2",
                cv2.CV_16S: "<i2",
                cv2.CV_32S: "<i4",
                cv2.CV_32F: "<f4",
                cv2.CV_64F: "<f8",
            }
            self.__cuda_array_interface__ = {
                "version": 3,
                "shape": (h, w, gpu_mat.channels())
                if gpu_mat.channels() > 1
                else (h, w),
                "typestr": type_map[gpu_mat.depth()],
                "descr": [("", type_map[gpu_mat.depth()])],
                "stream": 1,
                "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1())
                if gpu_mat.channels() > 1
                else (gpu_mat.step, gpu_mat.elemSize()),
                "data": (gpu_mat.cudaPtr(), False),
            }

    arr = cp.asarray(CudaArrayInterface(mat))
    return arr


def _hough_angles(pos, neg, orientation="H", thresh=(None, None)):
    height, width = pos.shape
    if orientation == "H":
        axis = 1
        theta = hough_theta_h
        fp = cp.ones((51, 1), dtype=cp.float32)
        margin = int(width * 0.975)
    elif orientation == "V":
        axis = 0
        theta = hough_theta_v
        fp = cp.ones((1, 51), dtype=cp.float32)
        margin = int(height * 0.025)
    else:
        raise RuntimeError(f"Unknown orientation {orientation}!")
    blur = ndi.median_filter(neg, footprint=fp, mode="reflect", cval=0.0)
    sums = cp.apply_along_axis(lambda a: a.sum(), axis, blur)
    line = sums.argmax(0)
    wsz = max(windowsize, margin)

    # Grab a +/- WINDOW-SIZE strip for evaluation. We've already cropped out the margins.
    if orientation == "H":
        cropped = pos[max(line - wsz, 0) : min(line + wsz, height)]  # noqa: E203
    else:
        cropped = pos[
            :,
            max(line - wsz, 0) : min(line + wsz, width),  # noqa: E203
        ]

    # default footprint = scipy.ndimage.generate_binary_structure(image.ndim, 1)
    # array([[False, True, False],
    #        [ True, True,  True],
    #        [False, True, False]])
    # see https://github.com/scipy/scipy/issues/13991 : (grey_)dilation is faster than binary_dilation!

    edges = ndi.grey_dilation(
        canny(
            cropped,
            sigma=2.0,
            low_threshold=thresh[0],
            high_threshold=thresh[1],
            mode="reflect",
        ),
        footprint=ndi.generate_binary_structure(cropped.ndim, 1),
    )

    line_length = int(height * 0.25)
    lines = probabilistic_hough_line(
        cp.asnumpy(edges), line_length=line_length, line_gap=2, theta=theta
    )
    # img_cv2 = cv_cuda_gpumat_from_cp_array(edges_u8)
    # print(edges_u8)
    # print(edges_u8.__cuda_array_interface__['data'][0])
    # print(img_cv2.download())
    # print(img_cv2.cudaPtr())
    # sadly, HoughLinesP is not in the cuda backend...
    # lines = cv2.HoughLinesP(img_cv2.download(),
    #                       1, #rho
    #                       theta,
    #                       10, #threshold
    #                       line_length, #minLineLength
    #                       2, #maxLineGap
    #                       )

    angles = []

    for (x0, y0), (x1, y1) in lines:
        # Ensure line is moving rightwards/upwards
        if orientation == "H":
            k = 1 if x1 > x0 else -1
            offset = 0
        else:
            k = 1 if y1 > y0 else -1
            offset = 90
        angles.append(
            (
                orientation,
                offset - np.rad2deg(math.atan2(k * (y1 - y0), k * (x1 - x0))),
                x0,
                y0,
                x1,
                y1,
            )
        )

    return (angles, edges)


def analyse_image(f, page, logger, pagenum=None):
    global debug, now
    if "debug" not in globals():
        debug = False
    if "now" not in globals():
        import datetime

        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    if "windowsize" not in globals():
        windowsize = 150

    pagenum = float(pagenum) if pagenum is not None else ""

    filename = os.path.basename(f)

    # This copies the image to the GPU
    page = cp.asarray(page, dtype=cp.float32)

    if page.ndim > 2:
        logger.debug(f"{filename} p{pagenum} - multichannel, converting to greyscale")
        page = rgb2gray(page)
    pageh, pagew = page.shape

    # Remove the margins, which are often dirty due to skewing
    # 0.33" of an 8.5" page is approximately 1/25th
    pos = crop(page, pagew // 25)

    if is_low_contrast(pos):
        logger.debug(f"{filename} p{pagenum} - low contrast - blank page?")
        return (f, pagenum, "", "", pagew, pageh)

    thr = 255 * 0.8
    p = np.clip(pos, 0, thr)
    neg = thr - p
    if debug:
        iio.imwrite(
            f"debug/{now}/{filename}_{pagenum}_neg.tiff",
            neg.get(),
        )

    # TODO: sanity check, _hough_angles should be under 100? 500?
    h_angles, h_edges = _hough_angles(pos, neg, "H")
    v_angles, v_edges = _hough_angles(pos, neg, "V")
    angles = h_angles + v_angles

    if len(angles) == 0:
        if debug:
            logger.debug(f"{filename} p{pagenum} - no lines found, changing threshold")
            iio.imwrite(
                f"debug/{now}/{filename}_{pagenum}_no_hlines.png",
                img_as_ubyte(_greyf(h_edges)).get(),
            )
            iio.imwrite(
                f"debug/{now}/{filename}_{pagenum}_no_vlines.png",
                img_as_ubyte(_greyf(v_edges)).get(),
            )
        h_angles, h_edges = _hough_angles(pos, neg, "H", thresh=(100, 255))
        v_angles, v_edges = _hough_angles(pos, neg, "V", thresh=(100, 255))
        angles = h_angles + v_angles
    else:
        if debug:
            iio.imwrite(
                f"debug/{now}/{filename}_{pagenum}_simple_dilation_h_edges.png",
                h_edges.get(),
            )
            iio.imwrite(
                f"debug/{now}/{filename}_{pagenum}_simple_dilation_v_edges.png",
                v_edges.get(),
            )

    if len(angles) > 0:
        angle = np.median([x[1] for x in angles])
        logger.debug(f"{filename} p{pagenum} - Hough angle median: {angle}°")
        if debug:
            hs = 0
            vs = 0
            h_edges_grey = _greyf(h_edges)
            v_edges_grey = _greyf(v_edges)
            for result in angles:
                orn, _, x0, y0, x1, y1 = result
                rr, cc, val = line_aa(c0=x0, r0=y0, c1=x1, r1=y1)
                if orn == "H":
                    hs += 1
                    for k, v in enumerate(val):
                        h_edges_grey[rr[k], cc[k]] = (1 - v) * h_edges_grey[
                            rr[k], cc[k]
                        ] + v
                else:  # V
                    vs += 1
                    for k, v in enumerate(val):
                        v_edges_grey[rr[k], cc[k]] = (1 - v) * v_edges_grey[
                            rr[k], cc[k]
                        ] + v
            if hs > 0:
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_hlines.png",
                    img_as_ubyte(h_edges_grey).get(),
                )
            if vs > 0:
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_vlines.png",
                    img_as_ubyte(v_edges_grey).get(),
                )
        # end if debug
        angles = [x[1] for x in angles]

    else:
        if debug:
            logger.debug(f"{filename} p{pagenum} - Failed simple Hough, dilating...")

        # We didn't find a good feature at the H or V sum peaks.
        # Let's brutally dilate everything and look for a vertical margin!

        height, width = neg.shape
        if height * width > int(1e6):
            # note: small will be float64 here, so, slower on GPU
            small = downscale_local_mean(neg, (2, 2))
        else:
            small = neg

        # performance wise, binary_dilation was incredibly bad.
        # see   https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4020#issuecomment-1409335950
        #   and https://scikit-image.org/docs/stable/auto_examples/applications/plot_rank_filters.html
        #   and https://github.com/scikit-image/scikit-image/issues/1190#issuecomment-57939688
        #       (ndimage and opencv faster than skimage.filter.rank.maximum)
        #   and the discussion here, decomposition is automatic for squares or rectangles:
        #       https://github.com/scipy/scipy/issues/13991#issuecomment-839954147
        #       https://github.com/scipy/scipy/blob/main/scipy/ndimage/_filters.py#L1723-L1730
        # pick a threshold for dilation
        thresh = threshold_otsu(small)
        dilated = ndi.grey_dilation(small > thresh, footprint=cp.ones((60, 60)))
        edges = canny(dilated, sigma=3.0)
        if debug:
            edges_grey = _greyf(edges)

        lines = probabilistic_hough_line(
            cp.asnumpy(edges),
            line_length=int(pageh * 0.04),
            line_gap=6,
            theta=hough_theta_hv,
        )

        for (x_0, y_0), (x_1, y_1) in lines:
            if abs(x_1 - x_0) > abs(y_1 - y_0):
                # angle is <= π/4 from horizontal or vertical
                _, x0, y0, x1, y1 = "H", x_0, y_0, x_1, y_1
            else:
                _, x0, y0, x1, y1 = "V", y_0, -x_0, y_1, -x_1
            # flip angle so that X delta is positive (East quadrants).
            k = 1 if x1 > x0 else -1
            a = np.rad2deg(math.atan2(k * (y1 - y0), k * (x1 - x0)))

            # Zero angles are suspicious -- could be a cropping margin.
            # If not, they don't add information anyway.
            if a != 0:
                angles.append(-a)
            if debug:
                rr, cc, val = line_aa(c0=x_0, r0=y_0, c1=x_1, r1=y_1)
                for k, v in enumerate(val):
                    edges_grey[rr[k], cc[k]] = (1 - v) * edges_grey[rr[k], cc[k]] + v

        if angles:
            angle = np.median(angles)
            logger.debug(
                f"{filename} p{pagenum} - dilated Hough angle median: {angle}°"
            )
            if debug:
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilated.png",
                    img_as_bool(dilated).get(),
                )
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_{angle}_lines.png",
                    img_as_ubyte(edges_grey).get(),
                )
        else:
            angle = None
            logger.debug(f"{filename} p{pagenum} - Failed dilated Hough, giving up")
            if debug:
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilated.png",
                    img_as_bool(dilated).get(),
                )
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilate_edges_grey.png",
                    img_as_ubyte(edges_grey).get(),
                )
                iio.imwrite(
                    f"debug/{now}/{filename}_{pagenum}_dilate_edges.png",
                    img_as_bool(edges).get(),
                )

    return (
        f,
        pagenum,
        angle if angle is not None else "",
        np.var(angles) if angles else "",
        pagew,
        pageh,
    )


# TODO: refactor everything below here into a new file
def _init_worker(queue, debug_arg, now_arg, wsz_arg):  # pragma: no cover
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    log_utils.setup_queue_logging(queue)
    # global is only in the multiprocessing Pool workers, not in main process.
    global debug, now, windowsize
    debug = debug_arg
    now = now_arg
    windowsize = wsz_arg


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
        pdf = pymupdf.open(f)
        return [(f, kind.mime, x) for x in range(len(pdf))]
    # TODO: Add support for multi-page TIFFs here via
    #   https://imageio.readthedocs.io/en/stable/reference/userapi.html#reading-images
    #   https://imageio.readthedocs.io/en/stable/userapi.html#imageio.mimread
    else:
        return [(f, kind.mime, None)]


def analyse_page(tuple):
    cp.cuda.set_pinned_memory_allocator(None)
    f, mimetype, page = tuple
    logger = logging.getLogger("hough")
    if page is None:
        # not a known multi-image file
        try:
            image = iio.imread(f)
            logger.info(f"Processing {f}...")
            return [analyse_image(f, image, logger)]
        except ValueError as e:
            # Fall through; might be multi-page anyway...
            logger.debug(f"Single-page read of {f} failed: {e}")
            if debug:
                print(traceback.format_exc())
        except OSError as e:
            # imageio v3 returns OSError if the type is unknown
            if "Could not find a backend" in str(e):
                logger.debug(f"Read of {f} failed: {e}")
                if debug:
                    print(traceback.format_exc())
    if mimetype == "application/pdf":
        results = []
        doc = pymupdf.open(f)
        imagelist = doc.get_page_images(page)
        for item in imagelist:
            # TODO: Add per-image timer
            xref = item[0]
            smask = item[1]
            if smask == 0:
                # TODO: handle imgdict == None
                imgdict = doc.extract_image(xref)
                # TODO: store and use imgdict.{xres|yres}
                # TODO: use imgdict.ext to inform iio.imread
                pagenum = float(f"{page + 1}.{xref}")
                logger.info(f"Processing {f} - page {pagenum}...")
                try:
                    image = iio.imread(imgdict["image"])
                    results.append(analyse_image(f, image, logger, pagenum=pagenum))
                except ValueError as e:
                    logger.error(f"Skipping {f} - page {pagenum}: {e}")
                    print(traceback.format_exc())
            else:
                logger.error(f"Skipping process {f} - page {pagenum} (smask=={smask})")
        return results
    else:
        # TODO: support multi-image TIFF with
        #   https://imageio.readthedocs.io/en/stable/reference/userapi.html#reading-images
        #   https://imageio.readthedocs.io/en/stable/userapi.html#imageio.mimread
        logger.error(f"Cannot process {f}: unknown file format")
        return []
