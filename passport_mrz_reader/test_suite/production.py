import os
from typing import Optional

import numpy as np
from PIL import Image

from passport_mrz_reader.common.engines import DeepLearning, Tesseract
from passport_mrz_reader.common.interfaces import PreProcessors, PostProcessors
from passport_mrz_reader.common.process import process

result: Optional[str] = process(
    np.asarray(
        Image.open(
            os.path.dirname(__file__) + "/../../data/images/PRADO MRZ/31028.jpeg"
        )
    ),
    PreProcessors(variable_threshold=True),
    Tesseract({}),
    PostProcessors(character_height=True, mrz_fields=True, line_lengths=True)
)

print(result)
