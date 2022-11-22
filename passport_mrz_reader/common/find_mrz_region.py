"""This module is used to find the MRZ region in a passport image."""

import numpy as np
import cv2
from PIL import Image
import imutils

from mrz_common import display_if_verbose, print_if_verbose


def find_mrz_region(image_url, verbose=False):
    """Find the region of the image that contains the MRZ.
    Assumes that the image is a passport with two MRZ lines with
    44 characters each.
    Arg:
        image_url: URL of the image
        verbose: Whether to print debug information and display images
    """
    image = np.asarray(Image.open(image_url))
    display_if_verbose("Original image", Image.fromarray(image), verbose)
    image = imutils.resize(image, width=1200)
    display_if_verbose("Resized image", Image.fromarray(image), verbose)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # change to binary image, set threshold according to the darkest
    # area of the image
    threshold = np.percentile(gray, 1.5)
    gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    display_if_verbose("Binary image", Image.fromarray(gray), verbose)
    (H, W) = gray.shape

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    # Smooth the image using a 3x3 Gaussian blur and then apply a
    # blackhat morpholigical operator to find dark regions on a light
    # background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    display_if_verbose("Blackhat image", Image.fromarray(blackhat), verbose)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (min_val, max_val) = (np.min(grad), np.max(grad))
    grad = (grad - min_val) / (max_val - min_val)
    grad = (grad * 255).astype("uint8")
    display_if_verbose("After min max scaling", Image.fromarray(grad), verbose)

    # Apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_kernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    display_if_verbose("Rect close", Image.fromarray(thresh), verbose)

    # find contours in the thresholded image and sort them from bottom
    # to top (since the MRZ will always be at the bottom of the passport)
    # The MRZ lines will be the two bottom contours
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = cv2.sort_contours(contours, method="bottom-to-top")[0]
    mrz_boxes = []
    for contour in contours:
        # compute the bounding box of the contour and then derive the
        # how much of the image the bounding box occupies in terms of
        # both width and height
        (x, y, w, h) = cv2.boundingRect(contour)
        percent_width = w / float(W)
        # percentHeight = h / float(H)
        # assume the line occupies at least 70% of the image width
        if percent_width > 0.7:
            mrz_box = (x, y, w, h)
            mrz_boxes.append(mrz_box)
        if len(mrz_boxes) == 2:
            break
    # if not both MRZ lines were found, return
    if len(mrz_boxes) < 2:
        print_if_verbose("MRZ could not be found", verbose)
        return None
    # pad the bounding box since we applied erosions and now need to
    # re-grow it
    mrz_lines = []
    for (x, y, w, h) in mrz_boxes:
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))
        # extract the padded MRZ from the image
        mrz_line = image[y : y + h, x : x + w]
        mrz_lines.append(mrz_line)
    # fit to common witdh
    max_width = max(mrz_line.shape[1] for mrz_line in mrz_lines)
    try:
        mrz_lines = [
            imutils.resize(mrz_line, width=max_width) for mrz_line in mrz_lines
        ][::-1]
    except (ZeroDivisionError, cv2.error):
        # One of the witdhs is 0 or max_width is None
        print_if_verbose("MRZ could not be found", verbose)
        return None
    # stack images
    mrz_region = np.vstack(mrz_lines)
    display_if_verbose("Whole MRZ region", Image.fromarray(mrz_region))
    Image.fromarray(mrz_region).save(f"MRZPART/{image_url.split('/')[-1]}")
    return mrz_region
