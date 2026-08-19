"""
Módulo de preprocesamiento de audio y texto para ClonVoz.
"""

from src.preprocessing.audio_loader import (
    AudioSample,
    AudioValidationError,
    load_reference_audio,
    save_audio_pcm,
)
from src.preprocessing.text_processor import (
    convert_to_rioplatense,
    segment_script,
)

__all__ = [
    "AudioSample",
    "AudioValidationError",
    "load_reference_audio",
    "save_audio_pcm",
    "convert_to_rioplatense",
    "segment_script",
]
