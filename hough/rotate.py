"""
Worker functions for a parallelizable deskewer.
"""
import logging
import os

import filetype
import fitz
import imageio.v3 as iio
from scipy import ndimage


def rotate(imagelist, out, generator=False):
    """Actually rotates a single file of 1+ images.
    Rotated file has the same name and is placed under the {out} subdirectory
    of the cwd.
    """
    logger = logging.getLogger("hough")
    guess = filetype.guess(imagelist[0][0])
    if guess and guess.mime == "application/pdf":
        newdoc = fitz.open()
    else:
        newdoc = None
    filename = imagelist[0][0]
    filen, ext = os.path.splitext(os.path.basename(filename))
    kind = filetype.guess(filename)
    for image in imagelist:
        page = int(image[1]) if image[1] else ""
        angle = float(image[2]) if image[2] else 0.0
        if not page:
            # single-image file, not a container
            logger.info(f"Rotating {filename}...")
            img = iio.imread(image[0])
            fixed = ndimage.rotate(img, -angle, mode="nearest", reshape=False)
            iio.imwrite(f"{out}/{filen}{ext}", fixed)
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
                            img = iio.imread(imgdict["image"])
                            imgext = imgdict["ext"]
                            fixed = ndimage.rotate(
                                img, -angle, mode="nearest", reshape=False
                            )
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
                                f"Skipping rotating {filename} - "
                                f"page {page} - xref {xref}: {e}"
                            )
                    else:
                        logger.error(
                            f"Skipping process {filename} - page {page} - "
                            f"image {xref} (smask=={smask})"
                        )
            # TODO: deal with other multi-image formats
            else:
                logger.error(
                    f"Skipping file {filename} - unknown multi-page file format"
                )
        yield 1
    if newdoc:
        newdoc.save(f"{out}/{filen}{ext}")
