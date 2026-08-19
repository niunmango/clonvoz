"""
Prueba integral de integración de pipeline completo (Carga -> Validación -> Síntesis -> Exportación a 48kHz PCM).
"""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from src.engine.voxcpm2_engine import VoxCPM2Engine
from src.preprocessing.audio_loader import load_reference_audio, save_audio_pcm
from src.preprocessing.text_processor import segment_script


def test_full_pipeline_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Crear audio de referencia de 11s a 44100 Hz
        ref_path = os.path.join(tmpdir, "ref_11s.wav")
        sr_orig = 44100
        dur_orig = 11.0
        t = np.linspace(0, dur_orig, int(sr_orig * dur_orig), endpoint=False, dtype=np.float32)
        sf.write(ref_path, 0.4 * np.sin(2 * np.pi * 440 * t), sr_orig)

        transcript = "Muestra de prueba de once segundos con audio limpio."

        # 2. Cargar y validar audio de referencia
        sample = load_reference_audio(
            audio_path=ref_path,
            transcript=transcript,
            min_duration=10.0,
            max_duration=15.0,
            target_sr=48000,
        )

        assert sample.sample_rate == 48000
        assert 10.0 <= sample.duration <= 15.0

        # 3. Preparar guion
        guion = "Primer bloque de síntesis para el pipeline.\n\nSegundo bloque con cierre del podcast."
        bloques = segment_script(guion)
        assert len(bloques) == 2

        # 4. Inferencia con VoxCPM2
        engine = VoxCPM2Engine()
        engine.load_model()

        audios = []
        for bloque in bloques:
            res = engine.synthesize(text=bloque, reference_audio=sample)
            assert res.sample_rate == 48000
            audios.append(res.audio)

        # 5. Concatenación y guardado
        audio_total = np.concatenate(audios)
        out_path = os.path.join(tmpdir, "podcast_final_48k.wav")
        save_audio_pcm(out_path, audio_total, sample_rate=48000, bit_depth="PCM_16")

        # 6. Verificación de archivo resultante
        assert os.path.exists(out_path)
        info = sf.info(out_path)
        assert info.samplerate == 48000
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.duration > 0
