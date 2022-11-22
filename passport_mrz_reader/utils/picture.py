"""
Module for removing colours from CV2 pictures
"""
import copy
import cv2
from cv2 import Mat

def color_diff_max(c_1: int, c_2: int, c_3: int) -> float:
    """
    Returns biggest difference between three colors
    """
    return max(abs(c_1 - (c_2 + c_3) / 2), abs(c_2 - (c_1 + c_3) / 2), abs(c_3 - (c_2 + c_1) / 2))


def image_rem_color(im_frame: Mat, variance: int = 17, b_trs: int = 80) -> Mat:
    """
    @input: frame (Matrix), variance: int, black_tresh: int
    returns a matrix with less colour
    """

    frame = copy.copy(im_frame)

    if frame is None:
        raise Exception("'Error loading image', Not valid frame")

    blue, green, red = cv2.split(frame)
    rows = frame.shape[0]
    cols = frame.shape[1]

    for i in range(rows):
        for j in range(cols):
            blue_frame = int(blue[i][j])
            green_frame = int(green[i][j])
            red_frame = int(red[i][j])

            check_1 = color_diff_max(red_frame, green_frame, blue_frame) > variance
            if check_1  and (green_frame > b_trs or red_frame > b_trs or blue_frame > b_trs):
                frame[i][j] = [240, 240, 240]

    return frame
