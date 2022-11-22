"""Process an image into text from start to finish"""
from typing import Optional

from PIL import Image

from passport_mrz_reader.common.interfaces import (
    PreProcessors,
    PostProcessors,
    Engine,
)
from passport_mrz_reader.common.mrz_common import (
    display_if_verbose,
    print_if_verbose,
)
from passport_mrz_reader.common.postprocessing import postprocess
from passport_mrz_reader.common.preprocessing import preprocess


def process(
    image,
    preprocessors: PreProcessors,
    engine: Engine,
    postprocessors: PostProcessors,
    verbose=False,
) -> Optional[str]:
    display_if_verbose(
        "Original image", Image.fromarray(image), verbose=verbose
    )
    if (
        preprocessors is not None
        and preprocessors.variable_threshold
        and (preprocessors.threshold or preprocessors.grayscale)
    ):
        print_if_verbose(
            "Only using variable threshold, disregarding threshold and grayscale",
            verbose=verbose,
        )
    # Pre-process
    if (
        preprocessors is not None
        and preprocessors.variable_threshold is not None
    ):
        pre_processed = None
    else:
        pre_processed = preprocess(image, preprocessors, verbose=verbose)
    original_image = image
    # Engine
    initial_result = engine.get_mrz_text(
        original_image, pre_processed, verbose=verbose
    )
    if initial_result is None:
        return None
    mrz_text, metadata = initial_result
    # Post-process
    post_processed = postprocess(
        mrz_text, metadata, postprocessors, verbose=verbose
    )
    return post_processed
