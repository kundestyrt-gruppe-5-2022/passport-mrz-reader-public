"""In this module Tesseract is used to predict the MRZ"""
from typing import Optional

import cv2
import pytesseract
from PIL import Image

from passport_mrz_reader.common.interfaces import (
    PostProcessorMetadata,
    PreProcessors,
)
from passport_mrz_reader.common.preprocessing import preprocess
from passport_mrz_reader.common.mrz_common import (
    MRZ_CHARACTERS,
    print_if_verbose,
    display_if_verbose,
)

# Config Tesseract to use MRZ characters only and to use the mrz language.
# To use the mrz language, the mrz.traineddata file must be in the tessdata
# folder where Tesseract is installed. Tesseract should be added to the PATH.
TESSERACT_CONFIG = f"-l mrz --psm 6 -c tessedit_char_whitelist={MRZ_CHARACTERS}"


def get_raw_mrz_text(
    original_image, preprocessed_image, verbose=False
) -> Optional[tuple[str, PostProcessorMetadata]]:
    """Get raw MRZ text from the MRZ region using Teserract.
    Boxes that have wrong proportions or overlap with other boxes are removed.
    Assumes that the image is from a passport with two MRZ lines
    with 44 characters each.

    Args:
        mrz_region: The region of the image that contains the MRZ
        verbose: Whether to print debug information and display images
    """
    if original_image is None and preprocessed_image is None:
        return None
    box_heights = []
    variable_threshold = preprocessed_image is None
    threshold_values = [10, 8, 12, 6, 14] if variable_threshold else [10]
    # OCR the MRZ region using Tesseract, only looking for valid MRZ characters
    i = 0
    while len(box_heights) != 88:
        if i == len(threshold_values):
            print_if_verbose(
                f"Wrong amount of boxes found, found {len(box_heights)} boxes",
                verbose,
            )
            return None
        if variable_threshold:
            mrz_region = preprocess(
                original_image,
                PreProcessors(grayscale=True, threshold=threshold_values[i]),
                verbose=verbose,
            )
        else:
            mrz_region = preprocessed_image
        try:
            boxes = pytesseract.image_to_boxes(mrz_region, config=TESSERACT_CONFIG)
        except ValueError:
            # mrz region is outside image
            return None
        mrz_letters = []
        all_boxes = mrz_region.copy()
        reduced_boxes = mrz_region.copy()
        box_heights = []
        for index, box in enumerate(boxes.splitlines()):
            box = box.split(" ")
            # draw the bounding box on the image
            H = all_boxes.shape[0]
            (x, y, w, h) = (
                int(box[1]),
                H - int(box[2]),
                int(box[3]),
                H - int(box[4]),
            )
            height, width = y - h, w - x
            cv2.rectangle(all_boxes, (x, y), (w, h), (0, 255, 0), 2)
            if 0 < index < len(boxes.splitlines()) - 1:
                previous = boxes.splitlines()[index - 1].split(" ")
                next_box = boxes.splitlines()[index + 1].split(" ")
                # Add newline if the box is on a new line
                if h > H - int(previous[2]):
                    mrz_letters.append("\n")
                # Discard if box is high and overlaps with one of the other boxes
                if (
                    height / width > 1.5
                    and not h > H - int(previous[2])
                    and not y < H - int(next_box[4])
                    and (x <= int(previous[3]) or x >= int(next_box[1]))
                ):
                    continue
                # Discard box if it overlaps with the previous and next box
                if x <= int(previous[3]) and w >= int(next_box[1]):
                    continue
            cv2.rectangle(reduced_boxes, (x, y), (w, h), (0, 255, 0), 2)
            mrz_letters.append(box[0])
            box_heights.append(height)
        i += 1
        display_if_verbose("All boxes", Image.fromarray(all_boxes), verbose)
        display_if_verbose("Reduced boxes", Image.fromarray(reduced_boxes), verbose)
    mrz_text = "".join(mrz_letters)
    print_if_verbose(f"MRZ text before postprocessing:\n{mrz_text}", verbose)
    return mrz_text, PostProcessorMetadata(box_heights=box_heights)
