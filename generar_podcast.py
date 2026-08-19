"""
Script principal de generación de podcast adaptado a VoxCPM2 (2B) y Nano-vLLM a 48 kHz.
Mantiene compatibilidad con ejecuciones directas ('python generar_podcast.py' / './generar.sh').
"""

import logging
import os
import shutil
import subprocess
import sys
import numpy as np

from src.config import DEFAULT_CONFIG, VoxCPM2Config
from src.engine.voxcpm2_engine import VoxCPM2Engine
from src.preprocessing.audio_loader import (
    AudioValidationError,
    load_reference_audio,
    save_audio_pcm,
)
from src.preprocessing.text_processor import segment_script

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("clonvoz")

# --- CONFIGURACIÓN DE ENTRADA / SALIDA ---
GUION_FILE = "guion.txt"
SAMPLE_TEXT_FILE = "sampleCorto.txt"
REF_AUDIO_FILE = "sampleCorto.wav"
OUTPUT_FILE = "podcast_completo.wav"


def normalizar_audio_ffmpeg(input_file: str, output_file: str, sample_rate: int = 48000) -> bool:
    """
    Normaliza el audio usando ffmpeg manteniendo 48 kHz PCM:
    1. Loudness Normalization a -14.0 LUFS (EBU R128 standard).
    2. Compressor para rango dinámico uniforme.
    3. Limiter de picos a -1.0dB.
    """
    if not shutil.which("ffmpeg"):
        logger.warning("⚠️ ffmpeg no encontrado. Se omite normalización avanzada de mastering.")
        return False

    temp_normalized = f"{os.path.splitext(input_file)[0]}.tmp.wav"
    filter_chain = "acompressor=threshold=-18dB:ratio=3:1:attack=20:release=250,loudnorm=I=-14:LRA=11:TP=-1.0"

    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-af", filter_chain,
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-y",
        temp_normalized
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
        if res.returncode == 0:
            os.replace(temp_normalized, output_file)
            logger.info(f"🎚️ Audio normalizado con FFmpeg a -14 LUFS (48 kHz) -> {output_file}")
            return True
        else:
            if os.path.exists(temp_normalized):
                os.remove(temp_normalized)
            return False
    except Exception as e:
        logger.warning(f"Error durante normalización FFmpeg: {e}")
        if os.path.exists(temp_normalized):
            os.remove(temp_normalized)
        return False


def main():
    logger.info("🎙️ ========================================================")
    logger.info("🎙️ ClonVoz 2.0: Síntesis Neural VoxCPM2 (2B) con Nano-vLLM")
    logger.info("🎙️ Frecuencia nativa: 48000 Hz | Soporte Rioplatense")
    logger.info("🎙️ ========================================================")

    # Validar archivos requeridos
    for f in [GUION_FILE, SAMPLE_TEXT_FILE, REF_AUDIO_FILE]:
        if not os.path.exists(f):
            logger.error(f"❌ Archivo requerido no encontrado: '{f}'")
            sys.exit(1)

    with open(SAMPLE_TEXT_FILE, "r", encoding="utf-8") as f:
        sample_transcript = f.read().strip()

    with open(GUION_FILE, "r", encoding="utf-8") as f:
        guion_raw = f.read()

    # 1. Validar muestra de audio [10.0s, 15.0s] con auto_trim para adaptarse
    try:
        ref_sample = load_reference_audio(
            audio_path=REF_AUDIO_FILE,
            transcript=sample_transcript,
            min_duration=DEFAULT_CONFIG.min_reference_duration,
            max_duration=DEFAULT_CONFIG.max_reference_duration,
            target_sr=48000,
            auto_trim=True,
        )
        logger.info(
            f"✅ Muestra de referencia cargada: {ref_sample.duration:.2f}s a 48 kHz "
            f"(Transcripción: '{ref_sample.transcript[:50]}...')"
        )
    except AudioValidationError as val_err:
        logger.error(f"❌ Error de validación en la muestra de audio: {val_err}")
        sys.exit(1)

    # 2. Inicializar motor VoxCPM2
    engine = VoxCPM2Engine(DEFAULT_CONFIG)
    engine.load_model()

    # 3. Segmentar guion
    bloques = segment_script(guion_raw)
    logger.info(f"📖 Guion preparado: {len(bloques)} bloques para síntesis.")

    os.makedirs(DEFAULT_CONFIG.temp_dir, exist_ok=True)
    segmentos_audio = []
    latencia_total = 0.0

    for i, bloque in enumerate(bloques, start=1):
        temp_file = os.path.join(DEFAULT_CONFIG.temp_dir, f"bloque_{i:02d}.wav")
        logger.info(f"⏳ Procesando bloque {i}/{len(bloques)} ({len(bloque)} chars)...")

        res = engine.synthesize(
            text=bloque,
            reference_audio=ref_sample,
            apply_rioplatense=True,
        )
        save_audio_pcm(temp_file, res.audio, sample_rate=48000, bit_depth="PCM_16")
        segmentos_audio.append(res.audio)
        latencia_total += res.latency_seconds

    # 4. Concatenación final a 48 kHz
    silencio_intermedio = np.zeros(int(48000 * 0.4), dtype=np.float32)
    piezas = [np.zeros(int(48000 * 0.5), dtype=np.float32)]  # Silencio inicial 0.5s
    for a in segmentos_audio:
        piezas.append(a)
        piezas.append(silencio_intermedio)

    audio_unificado = np.concatenate(piezas)
    duracion_total = len(audio_unificado) / 48000
    rtf_global = latencia_total / duracion_total if duracion_total > 0 else 0.0

    save_audio_pcm(OUTPUT_FILE, audio_unificado, sample_rate=48000, bit_depth="PCM_16")

    # Intentar normalización mastering FFmpeg
    normalizar_audio_ffmpeg(OUTPUT_FILE, OUTPUT_FILE, sample_rate=48000)

    # Limpieza temporal
    shutil.rmtree(DEFAULT_CONFIG.temp_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(f"✅ ¡Podcast completo generado con éxito!")
    logger.info(f"📁 Salida: {OUTPUT_FILE} (48000 Hz, 16-bit PCM)")
    logger.info(f"⏱️ Duración audio: {duracion_total:.2f}s | Tiempo inferencia: {latencia_total:.2f}s")
    logger.info(f"⚡ RTF global: {rtf_global:.4f} (Target <= {DEFAULT_CONFIG.target_rtf})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
