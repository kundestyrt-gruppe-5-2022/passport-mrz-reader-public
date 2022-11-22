"""
The engines capable of doing the main OCR task
"""
from typing import TypedDict, Optional

from passport_mrz_reader.common.interfaces import PostProcessorMetadata, Engine
from passport_mrz_reader.pure_tesseract import tesseract_predict
from passport_mrz_reader.easy_ocr import easy_ocr_predict
from passport_mrz_reader.deep_learning import tensor_flow_predictor


class TesseractOptions(TypedDict):
    """Options for the Tesseract engine"""


class Tesseract(Engine):
    """OCR engine using Tesseract"""

    def __init__(self, options: TesseractOptions):
        self.options = options

    def get_mrz_text(
        self, original_image, preprocessed_image, verbose=False
    ) -> Optional[tuple[str, PostProcessorMetadata]]:
        """Get the raw MRZ text using Tesseract"""
        return tesseract_predict.get_raw_mrz_text(
            original_image, preprocessed_image, verbose=verbose
        )


class EasyOcrOptions(TypedDict):
    """Options for the EasyOcr engine"""


class EasyOcr(Engine):
    """OCR engine using EasyOcr"""

    def __init__(self, options: EasyOcrOptions):
        self.options = options

    def get_mrz_text(
        self, original_image, preprocessed_image, verbose=False
    ) -> Optional[tuple[str, PostProcessorMetadata]]:
        """Get the raw MRZ text using EasyOcr"""
        return easy_ocr_predict.get_raw_mrz_text(
            original_image, preprocessed_image, verbose=verbose
        )


class DeepLearningOptions(TypedDict):
    """Options for DeepLearning engine"""


class DeepLearning(Engine):
    """OCR engine using custom deeplearning model with TensorFlow"""

    def __init__(self, options: DeepLearningOptions):
        self.options = options

    def get_mrz_text(
        self, original_image, preprocessed_image, verbose=False
    ) -> Optional[tuple[str, PostProcessorMetadata]]:
        """Get the raw MRZ using TensorFlow deeplearning"""
        return tensor_flow_predictor.make_prediction(
            original_image, preprocessed_image, verbose
        )
