"""This module is used to preprocess the image before OCR"""
import cv2
import imutils
import numpy as np
from PIL import Image

from passport_mrz_reader.common.mrz_common import display_if_verbose
from passport_mrz_reader.common.interfaces import PreProcessors


def preprocess(image, preprocessors: PreProcessors, verbose=False):
    """Preprocesses the image to make it easier to read.

    Args:
        image: Image to preprocess
        preprocessors: The configured pre-processors to use
        verbose: Whether to print debug information and display images
    """
    # resize image
    image = imutils.resize(image, width=1200)
    display_if_verbose("After resizing", Image.fromarray(image), verbose)
    if preprocessors.grayscale is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_if_verbose(
            "After grayscaling", Image.fromarray(image), verbose
        )
    if preprocessors.threshold is not None:
        # change to binary image, set threshold according to the darkest
        # area of the image
        threshold = np.percentile(image, preprocessors.threshold)
        image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
        display_if_verbose(
            f"After thresholding with threshold {preprocessors.threshold}",
            Image.fromarray(image),
            verbose,
        )
    if preprocessors.grayscale is not None:
        # change to color image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        display_if_verbose(
            "After un-grayscaling", Image.fromarray(image), verbose
        )
    return image
