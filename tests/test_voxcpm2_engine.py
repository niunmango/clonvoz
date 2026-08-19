"""
Pruebas para el motor VoxCPM2 (2B), Nano-vLLM y Fallback a CPU.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.config import VoxCPM2Config
from src.engine.voxcpm2_engine import SynthesisResult, VoxCPM2Engine, get_optimal_device
from src.preprocessing.audio_loader import AudioSample, AudioValidationError


@pytest.fixture
def dummy_audio_sample():
    sr = 48000
    duration = 12.0
    audio = np.random.uniform(-0.4, 0.4, int(sr * duration)).astype(np.float32)
    return AudioSample(
        audio=audio,
        sample_rate=sr,
        duration=duration,
        transcript="Transcripción de prueba para el modelo VoxCPM2.",
    )


def test_voxcpm2_engine_initialization():
    """Prueba la configuración por defecto de VoxCPM2."""
    config = VoxCPM2Config(
        model_id="openbmb/VoxCPM2",
        sample_rate=48000,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )
    engine = VoxCPM2Engine(config)
    assert engine.sample_rate == 48000
    assert engine.model_id == "openbmb/VoxCPM2"
    assert engine.config.target_rtf <= 0.15


def test_voxcpm2_cpu_fallback_config():
    """Prueba que en modo CPU se desactive FlashAttention-2 y se use SDPA y float32."""
    config = VoxCPM2Config(
        device="cpu",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )
    engine = VoxCPM2Engine(config)
    assert engine.device == "cpu"
    assert engine.attn_implementation == "sdpa"
    # En CPU el tipo de datos seguro es float32
    if engine.torch_dtype is not None:
        import torch
        assert engine.torch_dtype == torch.float32


def test_voxcpm2_cpu_synthesis(dummy_audio_sample):
    """Prueba la síntesis completa y generación en CPU a 48 kHz."""
    config = VoxCPM2Config(device="cpu")
    engine = VoxCPM2Engine(config)
    engine.load_model()

    text = "Probando la síntesis ejecutada exclusivamente en CPU con fallback."
    result = engine.synthesize(text=text, reference_audio=dummy_audio_sample)

    assert isinstance(result, SynthesisResult)
    assert result.sample_rate == 48000
    assert result.duration_seconds > 0
    assert result.metrics["device"] == "cpu"
    assert result.metrics["attn_implementation"] == "sdpa"


def test_voxcpm2_synthesis_metrics(dummy_audio_sample):
    """Prueba la síntesis, cálculo de RTF y formato de salida."""
    engine = VoxCPM2Engine()
    engine.load_model()

    text = "Probando la síntesis con Nano-vLLM y bfloat16 a cuarenta y ocho kilo hertz."
    result = engine.synthesize(text=text, reference_audio=dummy_audio_sample)

    assert isinstance(result, SynthesisResult)
    assert result.sample_rate == 48000
    assert result.duration_seconds > 0
    assert result.latency_seconds > 0
    assert result.rtf > 0
    assert pytest.approx(result.rtf, rel=1e-2) == (result.latency_seconds / result.duration_seconds)
    assert "target_rtf" in result.metrics
    assert result.metrics["sample_rate"] == 48000


def test_voxcpm2_rejects_missing_transcript():
    """Prueba que el motor rechace solicitudes sin muestra de referencia válida."""
    engine = VoxCPM2Engine()
    engine.load_model()

    with pytest.raises(AudioValidationError):
        engine.synthesize(text="Texto", reference_audio=None)


def test_voxcpm2_dynamic_oom_fallback_to_cpu(dummy_audio_sample):
    """Prueba que un error de CUDA/OOM conmute el motor dinámicamente a CPU y reintente con éxito."""
    config = VoxCPM2Config(device="cuda:0")
    engine = VoxCPM2Engine(config)

    # Simular modelo que falla con OOM en GPU en el primer intento
    mock_model = MagicMock()
    mock_model.generate.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    engine.model = mock_model
    engine.is_loaded = True

    # Ejecutar síntesis: debe conmutar a CPU y resolver exitosamente
    res = engine.synthesize("Texto de prueba con OOM", reference_audio=dummy_audio_sample)

    assert engine.device == "cpu"
    assert res.sample_rate == 48000
    assert res.duration_seconds > 0
