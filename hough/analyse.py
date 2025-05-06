"""
Worker functions for a parallelizable skew analyser.
"""

import bisect
import csv
import math
from enum import Enum
from multiprocessing.pool import Pool
from pathlib import Path


try:  # pragma: no cover
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    from cucim.skimage.color import rgb2gray
    from cucim.skimage.exposure import is_low_contrast
    from cucim.skimage.feature import canny
    from cucim.skimage.filters import threshold_otsu
    from cucim.skimage.transform import downscale_local_mean
    from cucim.skimage.util import crop, img_as_bool, img_as_ubyte
except ModuleNotFoundError:  # pragma: no cover
    import numpy as cp
    import scipy.ndimage as ndi
    from skimage.color import rgb2gray
    from skimage.exposure import is_low_contrast
    from skimage.feature import canny
    from skimage.filters import threshold_otsu
    from skimage.transform import downscale_local_mean
    from skimage.util import crop, img_as_bool, img_as_ubyte

import imageio.v3 as iio
import numpy as np
from loguru import logger
from skimage.draw import line_aa
from skimage.transform import probabilistic_hough_line
from tqdm import tqdm  # TODO: consider rich?

from . import (
    CommonArgs,
    abort,
    get_images,
    get_num_images,
    get_cpu_image,
    get_gpu_image_maybe,
)


# TODO: configfile out all magic numbers, including windowsize
# TODO: class?

Orientation = Enum("Orientation", "HORIZONTAL,VERTICAL")


# hough's little helpers
def _greyf(arr):
    return (arr > 0.3) * 0.3


# CuPy scalars are actually GPU-resident zero-dimensional arrays, so we use numpy here instead
hough_prec = np.deg2rad(0.02)
hough_theta_h = np.arange(
    np.deg2rad(-93.0), np.deg2rad(-87.0), hough_prec, dtype=np.float64
)
hough_theta_v = np.arange(
    np.deg2rad(-3.0), np.deg2rad(3.0), hough_prec, dtype=np.float64
)
hough_theta_hv = np.concatenate((hough_theta_v, hough_theta_h))


def _hough_angles(pos, neg, orientation=Orientation.HORIZONTAL, thresh=(None, None)):
    height, width = pos.shape
    if orientation == Orientation.HORIZONTAL:
        axis = 1
        theta = hough_theta_h
        fp = cp.ones((51, 1), dtype=np.float32)
        margin = int(width * 0.975)
    elif orientation == Orientation.VERTICAL:
        axis = 0
        theta = hough_theta_v
        fp = cp.ones((1, 51), dtype=np.float32)
        margin = int(height * 0.025)
    else:  # pragma: no cover
        pass
    blur = ndi.median_filter(neg, footprint=fp, mode="reflect", cval=0.0)
    sums = cp.apply_along_axis(lambda a: a.sum(), axis, blur)
    line = sums.argmax(0)
    wsz = max(windowsize, margin)

    # Grab a +/- WINDOW-SIZE strip for evaluation. We've already cropped out the margins.
    if orientation == Orientation.HORIZONTAL:
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
        get_cpu_image(edges), line_length=line_length, line_gap=2, theta=theta
    )

    # TODO: try OpenCV HoughLinesP
    # img_cv2 = cv_cuda_gpumat_from_cp_array(edges_u8)
    # print(edges_u8)
    # print(edges_u8.__cuda_array_interface__['data'][0])
    # print(img_cv2.download())
    # print(img_cv2.cudaPtr())
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
        if orientation == Orientation.HORIZONTAL:
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


def analyse_image(
    image: tuple[
        Path,  # file
        tuple[int, int] | None,  # (pagenum, xref) if multi-image file
        tuple[int, int] | None,  # (xres, yres) dpi
        np.ndarray,  # image
    ],
):
    """Analyses an image for deskewing, potentially using a CUDA backend."""
    if "cuda" in dir(cp):  # pragma: no cover
        cp.cuda.set_pinned_memory_allocator(None)
    page = get_gpu_image_maybe(image[3])

    f = image[0]
    filename = f.name
    pagenum = float(f"{image[1][0]}.{image[1][1]}") if image[1] else 0.0

    global windowsize
    if "windowsize" not in globals():
        windowsize = 150

    if page.ndim > 2:
        logger.debug(f"{filename} p{pagenum} - multichannel, converting to greyscale")
        # discard any alpha channel
        page = rgb2gray(page[:, :, :3])
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
    if common.debug:
        dimg = get_cpu_image(neg)
        iio.imwrite(
            f"{common.debugpath}/{filename}_{pagenum}_neg.tiff",
            dimg,
        )

    # TODO: sanity check, _hough_angles should be under 100? 500?
    h_angles, h_edges = _hough_angles(pos, neg, Orientation.HORIZONTAL)
    v_angles, v_edges = _hough_angles(pos, neg, Orientation.VERTICAL)
    angles = h_angles + v_angles

    if len(angles) == 0:
        logger.debug(f"{filename} p{pagenum} - no lines found, changing threshold")
        if common.debug:
            dhedges = get_cpu_image(img_as_ubyte(_greyf(h_edges)))
            dvedges = get_cpu_image(img_as_ubyte(_greyf(v_edges)))
            iio.imwrite(
                f"{common.debugpath}/{filename}_{pagenum}_no_hlines.png",
                dhedges,
            )
            iio.imwrite(
                f"{common.debugpath}/{filename}_{pagenum}_no_vlines.png",
                dvedges,
            )
        h_angles, h_edges = _hough_angles(
            pos, neg, Orientation.HORIZONTAL, thresh=(100, 255)
        )
        v_angles, v_edges = _hough_angles(
            pos, neg, Orientation.VERTICAL, thresh=(100, 255)
        )
        angles = h_angles + v_angles
    else:
        if common.debug:
            dhedges = get_cpu_image(h_edges)
            dvedges = get_cpu_image(v_edges)
            iio.imwrite(
                f"{common.debugpath}/{filename}_{pagenum}_simple_dilation_h_edges.png",
                dhedges,
            )
            iio.imwrite(
                f"{common.debugpath}/{filename}_{pagenum}_simple_dilation_v_edges.png",
                dvedges,
            )

    if len(angles) > 0:
        angle = np.median([x[1] for x in angles])
        logger.debug(f"{filename} p{pagenum} - Hough angle median: {angle}°")
        if common.debug:
            hs = 0
            vs = 0
            h_edges_grey = _greyf(h_edges)
            v_edges_grey = _greyf(v_edges)
            for result in angles:
                orn, _, x0, y0, x1, y1 = result
                rr, cc, val = line_aa(c0=x0, r0=y0, c1=x1, r1=y1)
                if orn == Orientation.HORIZONTAL:
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
                dhg = get_cpu_image(img_as_ubyte(h_edges_grey))
                iio.imwrite(f"{common.debugpath}/{filename}_{pagenum}_hlines.png", dhg)
            if vs > 0:
                vhg = get_cpu_image(img_as_ubyte(v_edges_grey))
                iio.imwrite(f"{common.debugpath}/{filename}_{pagenum}_vlines.png", vhg)
        # end if common.debug
        angles = [x[1] for x in angles]

    else:
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
        if common.debug:
            edges_grey = _greyf(edges)

        npedges = get_cpu_image(edges)
        lines = probabilistic_hough_line(
            npedges,
            line_length=int(pageh * 0.04),
            line_gap=6,
            theta=hough_theta_hv,
        )

        for (x_0, y_0), (x_1, y_1) in lines:
            if abs(x_1 - x_0) > abs(y_1 - y_0):
                # angle is <= π/4 from horizontal or vertical
                _, x0, y0, x1, y1 = Orientation.HORIZONTAL, x_0, y_0, x_1, y_1
            else:
                _, x0, y0, x1, y1 = Orientation.VERTICAL, y_0, -x_0, y_1, -x_1
            # flip angle so that X delta is positive (East quadrants).
            k = 1 if x1 > x0 else -1
            a = np.rad2deg(math.atan2(k * (y1 - y0), k * (x1 - x0)))

            # Zero angles are suspicious -- could be a cropping margin.
            # If not, they don't add information anyway.
            if a != 0:
                angles.append(-a)
            if common.debug:
                rr, cc, val = line_aa(c0=x_0, r0=y_0, c1=x_1, r1=y_1)
                for k, v in enumerate(val):
                    edges_grey[rr[k], cc[k]] = (1 - v) * edges_grey[rr[k], cc[k]] + v

        if angles:
            angle = np.median(angles)
            logger.debug(
                f"{filename} p{pagenum} - dilated Hough angle median: {angle}°"
            )
            if common.debug:
                dd = get_cpu_image(img_as_bool(dilated))
                deg = get_cpu_image(img_as_ubyte(edges_grey))
                iio.imwrite(
                    f"{common.debugpath}/{filename}_{pagenum}_dilated.png",
                    dd,
                )
                iio.imwrite(
                    f"{common.debugpath}/{filename}_{pagenum}_{angle}_lines.png",
                    deg,
                )
        else:
            angle = None
            logger.debug(f"{filename} p{pagenum} - Failed dilated Hough, giving up")
            if common.debug:
                dd = get_cpu_image(img_as_bool(dilated))
                deg = get_cpu_image(img_as_ubyte(edges_grey))
                de = get_cpu_image(img_as_bool(edges))
                iio.imwrite(f"{common.debugpath}/{filename}_{pagenum}_dilated.png", dd)
                iio.imwrite(
                    f"{common.debugpath}/{filename}_{pagenum}_dilate_edges_grey.png",
                    deg,
                )
                iio.imwrite(
                    f"{common.debugpath}/{filename}_{pagenum}_dilate_edges.png", de
                )

    return (
        f,
        pagenum,
        angle if angle is not None else "",
        np.var(angles) if angles else "",
        pagew,
        pageh,
    )


# NOTE: _init_worker must be in the same module as analyse_files (i.e. the call to the initializer), not sure why
def _init_worker(common_arg: CommonArgs):
    #    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global common, logger
    common = common_arg
    logger = common_arg.logger


def analyse_files(files: list[Path], common: CommonArgs) -> Path:
    results = []

    with Pool(
        processes=common.workers, initializer=_init_worker, initargs=(common,)
    ) as p:
        try:
            with tqdm(
                total=get_num_images(files),
                disable=(common.loglevel != "WARNING"),
                unit="pg",
                desc="Analysis: ",
            ) as pbar:
                for i, result in enumerate(
                    p.imap_unordered(
                        analyse_image, get_images(files), min(common.workers, 8)
                    )
                ):
                    pbar.update()
                    bisect.insort(results, result, key=lambda x: x[1])
        except NotImplementedError as e:
            logger.error(e)
            abort(p)
            return None
        except KeyboardInterrupt:
            import sys

            print("Caught KeyboardInterrupt, terminating workers...", file=sys.stderr)
            abort(p)
            return None
        finally:
            # https://coverage.readthedocs.io/en/7.8.0/subprocess.html#using-multiprocessing
            p.close()
            p.join()

    # TODO: don't clobber existing results file?
    resultspath = Path(common.outpath, "results.csv")
    with open(resultspath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(
            [
                "Input File",
                "Page Number",
                "Computed angle",
                "Variance of computed angles",
                "Image width (px)",
                "Image height (px)",
                # "X pixels per inch",
                # "Y pixels per inch",
            ]
        )
        writer.writerows(results)
    return resultspath
