"""In this module EasyOCR is used to predict the MRZ of passports"""

from typing import Optional
import easyocr
from passport_mrz_reader.common.interfaces import (
    PostProcessorMetadata,
    PreProcessors,
)
from passport_mrz_reader.common.preprocessing import preprocess

from passport_mrz_reader.common.mrz_common import (
    MRZ_CHARACTERS,
    print_if_verbose,
)

# Load model
READER = easyocr.Reader(["en"], gpu=False, verbose=False)


def get_raw_mrz_text(
    original_image, preprocessed_image, verbose=False
) -> Optional[tuple[str, PostProcessorMetadata]]:
    """Get the raw MRZ text using EasyOCR

    Args:
        mrz_region(numpy.ndarray): The MRZ region of the passport
        verbose(bool): Whether to print verbose information
    """
    variable_threshold = preprocessed_image is None
    threshold_values = [10, 8, 12, 6, 14] if variable_threshold else [-1]
    i = 0
    result = None
    while result is None:
        if i == len(threshold_values):
            print_if_verbose(
                f"No result found",
                verbose,
            )
            return None
        preprocessed_image = (
            preprocess(
                original_image,
                PreProcessors(grayscale=True, threshold=threshold_values[i]),
                verbose=verbose,
            )
            if variable_threshold
            else preprocessed_image
        )
        result = READER.readtext(
            preprocessed_image, detail=0, allowlist=MRZ_CHARACTERS
        )
        if result:
            raw_mrz_text = "\n".join(result)
            print_if_verbose(f"Raw MRZ text:\n{raw_mrz_text}", verbose)
            return raw_mrz_text, PostProcessorMetadata()
    return None
