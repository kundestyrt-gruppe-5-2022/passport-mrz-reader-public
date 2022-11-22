"""Module of helper functions used to predict mrz field with TensorFlow deep learning model"""

import os
import sys
import pathlib
from typing import Optional

import cv2
import imutils
import numpy as np
from PIL import Image


import tensorflow as tf
from passport_mrz_reader.common.interfaces import (
    PostProcessorMetadata,
    PreProcessors,
)
from passport_mrz_reader.common.mrz_common import print_if_verbose

from passport_mrz_reader.custom_character_separator.custom_character_separator import (
    get_bounding_boxes,
)

VALUE_TO_LETTER = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "<",
}

# Load model
PROJECT_ROOT = f"{os.path.dirname(__file__)}/.."
MODEL_PATH = f"{PROJECT_ROOT}/deep_learning/final_model/3"
MODEL = tf.keras.models.load_model(MODEL_PATH)


def create_image_folder(original_image: Image, preprocessed_image, verbose=False):
    """Create folder for each character in an image"""

    # Create folder
    base_folder = f"{PROJECT_ROOT}/deep_learning/empty_temp"
    folder = os.path.join(base_folder, "temp_folder")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Add each character to folder as standalone image
    variable_threshold = preprocessed_image is None
    original_image = imutils.resize(original_image, width=1200)
    threshold_values = [10, 8, 12, 6, 14] if variable_threshold else [10]
    boxes = []
    index = 0
    while len(boxes) != 88:
        if index == len(threshold_values):
            print_if_verbose("Could not find 88 characters", verbose)
            return -1, None
        boxes, preprocessed_image = get_bounding_boxes(
            original_image,
            PreProcessors(grayscale=True, threshold=threshold_values[index]),
            verbose,
        )
        index += 1
    box_heights = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        box_heights.append(h)
        character_image = preprocessed_image[y - 1 : y + h + 1, x - 1 : x + w + 1]
        folder = f"{folder}"
        if i < 10:
            temp = f"0{i}"
        else:
            temp = i
        try:
            cv2.imwrite(f"{folder}/box_{temp}.jpeg", character_image)
        except cv2.error:
            continue
    return folder, box_heights


def format_dataset(path: pathlib.Path):
    """Format image to fit model input"""
    # Redirect stdout to avoid printing
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        image_size=(180, 180),
        batch_size=32,
        shuffle=False,
    )
    sys.stdout = stdout
    return dataset


def predict_letter(path: pathlib.Path, verbose=False):
    """Load model and predict letter in image from path"""

    tensor = format_dataset(path)
    predictions = MODEL.predict(tensor, verbose=verbose)

    # Remove temp files
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    os.rmdir(path)

    mrz = []
    # For debugging, possibly future preprocessing
    # index = 0
    for prediction in predictions:
        # For debugging, possibly future postprocessing
        # print(prediction)
        # print(np.argmax(prediction), tensor.file_paths[index])
        # index +=1
        mrz.append(VALUE_TO_LETTER[np.argmax(prediction)])
    mrz_text = "".join(mrz)
    mrz_text = f"{mrz_text[0:44]}\n{mrz_text[44:]}"
    print_if_verbose(f"MRZ text before postprocessing:\n{mrz_text}", verbose)
    return mrz_text


def make_prediction(
    original_image: Image, preprocessed_image: Image, verbose=False
) -> Optional[tuple[str, PostProcessorMetadata]]:
    """Make prediction for mrz, return mrz value"""
    folder, box_heights = create_image_folder(
        original_image, preprocessed_image, verbose
    )
    if folder == -1:
        print_if_verbose("An error ocurred", verbose)
        return None, PostProcessorMetadata()

    return predict_letter(folder, verbose), PostProcessorMetadata(
        box_heights=box_heights
    )
