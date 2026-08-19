"""
Pruebas de esquemas Pydantic y validaciones de API.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import PodcastRequest, SynthesisRequest, SynthesisResponse


def test_synthesis_request_valid():
    """Prueba que un esquema válido sea instanciado correctamente."""
    req = SynthesisRequest(
        text="Texto para sintetizar",
        sample_audio_path="sampleCorto.wav",
        sample_transcript="Transcripción exacta de la muestra.",
        auto_trim=True,
    )
    assert req.text == "Texto para sintetizar"
    assert req.sample_audio_path == "sampleCorto.wav"
    assert req.sample_transcript == "Transcripción exacta de la muestra."
    assert req.auto_trim is True


def test_synthesis_request_requires_transcript():
    """Prueba que la omisión o vacío de sample_transcript arroje ValidationError de Pydantic."""
    with pytest.raises(ValidationError):
        SynthesisRequest(
            text="Texto para sintetizar",
            sample_audio_path="sampleCorto.wav",
            sample_transcript="",  # Vacío
        )

    with pytest.raises(ValidationError):
        SynthesisRequest(
            text="Texto para sintetizar",
            sample_audio_path="sampleCorto.wav",
            sample_transcript="   ",  # Espacios
        )


def test_synthesis_request_requires_text():
    """Prueba que el texto a sintetizar sea obligatorio."""
    with pytest.raises(ValidationError):
        SynthesisRequest(
            text="",
            sample_audio_path="sampleCorto.wav",
            sample_transcript="Transcripción válida",
        )
