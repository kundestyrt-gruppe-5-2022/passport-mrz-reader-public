"""This module implements a custom model for separating characters in the MRZ region.
It creates bounding boxes for every character in the MRZ region which in turn can be
passed to another model for character recognition.
"""

from functools import cmp_to_key
import cv2
from PIL import Image
from IPython.display import display
from passport_mrz_reader.common.interfaces import PreProcessors

from passport_mrz_reader.common.mrz_common import (
    display_if_verbose,
    print_if_verbose,
)

from passport_mrz_reader.common.preprocessing import preprocess
from passport_mrz_reader.common.interfaces import PreProcessors


def draw_numerated_boxes(image, boxes):
    """Draws numerated boxes on the image.

    Args:
        image: The image to draw on
        boxes: The bounding boxes to draw
    """
    for index, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(index),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return image


def _sort_bounding_boxes(bounding_boxes):
    """Sort bounding boxes from left to right and top to bottom"""

    def compare_bounding_boxes(box1, box2):
        vertical_threshold = 30
        if box1[0] == box2[0] and box1[1] == box2[1]:
            return 0
        vertical_distance = box1[1] - box2[1]
        if vertical_distance > vertical_threshold:
            # box1 is on line 2 and box2 is on line 1
            return 1
        return (
            1
            if box1[0] > box2[0]
            and abs(vertical_distance) < vertical_threshold
            else -1
        )

    bounding_boxes.sort(key=cmp_to_key(compare_bounding_boxes))
    return bounding_boxes


def _merge_bounding_boxes(bounding_boxes):
    """Merge bounding boxes where a character is split"""
    index = 0
    while index < len(bounding_boxes) - 1:
        x1, y1, w1, h1 = bounding_boxes[index]
        if index == len(bounding_boxes) - 1:
            break
        x2, y2, w2, h2 = bounding_boxes[index + 1]
        if abs(y1 - y2) < h1 and abs(x1 - x2) < w1:
            bounding_boxes[index] = (
                min(x1, x2),
                min(y1, y2),
                max(x1 + w1, x2 + w2) - min(x1, x2),
                max(y1 + h1, y2 + h2) - min(y1, y2),
            )
            print(
                f"merging {bounding_boxes[index]} with {bounding_boxes[index+1]}",
                f"on index {index}",
            )
            del bounding_boxes[index + 1]
            continue
        index += 1
    return bounding_boxes


def get_bounding_boxes(
    mrz_region, preprocessors: PreProcessors, verbose=False
):
    """Gets the bounding boxes for every character in the MRZ region.

    Args:
        mrz_region: The MRZ region image
        verbose: Whether to print debug information and display images
    """
    # Change to binary image and invert colors
    mrz_region = preprocess(mrz_region, preprocessors, verbose)
    preprocessed = mrz_region.copy()
    original_image = mrz_region.copy()
    mrz_region = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    mrz_region = cv2.bitwise_not(mrz_region)
    display_if_verbose(
        "Inverted MRZ region", Image.fromarray(mrz_region), verbose
    )
    # Connect characters that are split
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mrz_region = cv2.dilate(mrz_region, kernel, iterations=1)
    # mrz_region = cv2.morphologyEx(mrz_region, cv2.MORPH_CLOSE, kernel)
    # mrz_region = cv2.erode(mrz_region, kernel, iterations=1)
    # display_if_verbose(
    #     "Connected split characters", Image.fromarray(mrz_region), verbose
    # )

    # Find separate bounding boxes for every character
    contours, _ = cv2.findContours(
        mrz_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Sort bounding boxes from left to right and top to bottom
    bounding_boxes = _sort_bounding_boxes(bounding_boxes)
    display_if_verbose(
        "Original bounding boxes",
        Image.fromarray(
            draw_numerated_boxes(original_image.copy(), bounding_boxes)
        ),
        verbose,
    )
    bounding_boxes_dropped = bounding_boxes.copy()

    # Drop bounding boxes that are too small or too large
    bounding_boxes_dropped = [
        bounding_box
        for bounding_box in bounding_boxes
        if 75 < bounding_box[2] * bounding_box[3] < 1000
        and bounding_box[3] > 15
        and bounding_box[2] > 5
    ]
    display_if_verbose(
        "After dropping small and large boxes",
        Image.fromarray(
            draw_numerated_boxes(original_image.copy(), bounding_boxes_dropped)
        ),
        verbose,
    )

    # # Merge bounding boxes where a character is split
    # bounding_boxes_dropped = _merge_bounding_boxes(bounding_boxes_dropped)
    # display_if_verbose(
    #     "After merging boxes",
    #     Image.fromarray(
    #         draw_numerated_boxes(
    #             original_image.copy(),
    #             bounding_boxes_dropped,
    #         )
    #     ),
    #     verbose,
    # )

    print_if_verbose(
        f"Found {len(bounding_boxes_dropped)} bounding boxes", verbose
    )
    return bounding_boxes_dropped, preprocessed
