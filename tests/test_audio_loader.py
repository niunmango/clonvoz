"""
Pruebas para el cargador y validador de audio a 48 kHz.
"""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from src.preprocessing.audio_loader import (
    AudioSample,
    AudioValidationError,
    load_reference_audio,
    resample_audio,
    save_audio_pcm,
    to_mono,
)


@pytest.fixture
def temp_audio_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_dummy_wav(path: str, duration_s: float, sample_rate: int = 44100, channels: int = 1):
    """Crea un archivo WAV sintético para pruebas."""
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
    if channels == 1:
        data = 0.5 * np.sin(2 * np.pi * 440 * t)
    else:
        channel_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        data = np.stack([channel_data] * channels, axis=-1)

    sf.write(path, data, sample_rate)
    return path


def test_audio_loader_valid_12s(temp_audio_dir):
    """Prueba la carga exitosa de una muestra válida de 12 segundos."""
    wav_path = os.path.join(temp_audio_dir, "valid_12s.wav")
    create_dummy_wav(wav_path, duration_s=12.0, sample_rate=44100, channels=2)

    transcript = "Esta es una muestra de audio válida para clonado de voz."
    sample = load_reference_audio(
        audio_path=wav_path,
        transcript=transcript,
        min_duration=10.0,
        max_duration=15.0,
        target_sr=48000,
    )

    assert isinstance(sample, AudioSample)
    assert sample.sample_rate == 48000
    assert pytest.approx(sample.duration, rel=1e-2) == 12.0
    assert sample.transcript == transcript
    assert sample.audio.ndim == 1  # Debe ser mono


def test_audio_loader_rejects_under_10s(temp_audio_dir):
    """Prueba que se rechacen muestras menores a 10 segundos."""
    wav_path = os.path.join(temp_audio_dir, "short_5s.wav")
    create_dummy_wav(wav_path, duration_s=5.0, sample_rate=48000)

    with pytest.raises(AudioValidationError) as excinfo:
        load_reference_audio(
            audio_path=wav_path,
            transcript="Texto de prueba",
            min_duration=10.0,
            max_duration=15.0,
        )
    assert "below minimum" in str(excinfo.value)


def test_audio_loader_rejects_over_15s_strict(temp_audio_dir):
    """Prueba que se rechacen muestras mayores a 15 segundos sin auto_trim."""
    wav_path = os.path.join(temp_audio_dir, "long_20s.wav")
    create_dummy_wav(wav_path, duration_s=20.0, sample_rate=48000)

    with pytest.raises(AudioValidationError) as excinfo:
        load_reference_audio(
            audio_path=wav_path,
            transcript="Texto de prueba",
            min_duration=10.0,
            max_duration=15.0,
            auto_trim=False,
        )
    assert "exceeds maximum" in str(excinfo.value)


def test_audio_loader_auto_trim_over_15s_by_default(temp_audio_dir):
    """Prueba que por defecto (sin especificar auto_trim) se recorte a 15.0s."""
    wav_path = os.path.join(temp_audio_dir, "long_22s.wav")
    create_dummy_wav(wav_path, duration_s=22.0, sample_rate=48000)

    sample = load_reference_audio(
        audio_path=wav_path,
        transcript="Texto de prueba",
        min_duration=10.0,
        max_duration=15.0,
    )

    assert sample.duration == 15.0
    assert len(sample.audio) == int(15.0 * 48000)


def test_audio_loader_requires_transcript(temp_audio_dir):
    """Prueba que la transcripción sea estrictamente obligatoria."""
    wav_path = os.path.join(temp_audio_dir, "valid_12s.wav")
    create_dummy_wav(wav_path, duration_s=12.0, sample_rate=48000)

    with pytest.raises(AudioValidationError) as exc_none:
        load_reference_audio(wav_path, transcript=None)
    assert "Reference transcript is required" in str(exc_none.value)

    with pytest.raises(AudioValidationError) as exc_empty:
        load_reference_audio(wav_path, transcript="   ")
    assert "Reference transcript is required" in str(exc_empty.value)


def test_save_audio_pcm_metadata(temp_audio_dir):
    """Prueba que el archivo guardado tenga metadata exacta de 48 kHz PCM."""
    out_file = os.path.join(temp_audio_dir, "output_48k.wav")
    data = np.random.uniform(-0.5, 0.5, 48000 * 3).astype(np.float32)

    save_audio_pcm(out_file, data, sample_rate=48000, bit_depth="PCM_16")

    info = sf.info(out_file)
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.format == "WAV"
    assert pytest.approx(info.duration, rel=1e-2) == 3.0
