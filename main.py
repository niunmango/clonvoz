"""
Punto de entrada principal CLI para ClonVoz (VoxCPM2 + Nano-vLLM).
"""

import argparse
import logging
import os
import sys
import time
import numpy as np

from src.config import DEFAULT_CONFIG, VoxCPM2Config
from src.engine.voxcpm2_engine import VoxCPM2Engine, get_optimal_device
from src.preprocessing.audio_loader import (
    AudioSample,
    AudioValidationError,
    load_reference_audio,
    save_audio_pcm,
)
from src.preprocessing.text_processor import convert_to_rioplatense, segment_script

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("clonvoz.main")


def cmd_generate(args):
    """Genera el podcast completo por bloques a 48 kHz."""
    logger.info("🎙️ Iniciando generación de podcast con VoxCPM2 (2B) a 48 kHz...")

    # Validar archivos
    if not os.path.exists(args.guion):
        logger.error(f"❌ No se encontró el archivo de guion '{args.guion}'")
        sys.exit(1)
    if not os.path.exists(args.audio_ref):
        logger.error(f"❌ No se encontró el audio de referencia '{args.audio_ref}'")
        sys.exit(1)
    if not os.path.exists(args.transcript_ref):
        logger.error(f"❌ No se encontró la transcripción de referencia '{args.transcript_ref}'")
        sys.exit(1)

    with open(args.transcript_ref, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    with open(args.guion, "r", encoding="utf-8") as f:
        guion_raw = f.read()

    # 1. Validar y cargar muestra de referencia [10.0s, 15.0s]
    try:
        ref_sample = load_reference_audio(
            audio_path=args.audio_ref,
            transcript=transcript,
            min_duration=DEFAULT_CONFIG.min_reference_duration,
            max_duration=DEFAULT_CONFIG.max_reference_duration,
            target_sr=DEFAULT_CONFIG.sample_rate,
            auto_trim=args.auto_trim,
        )
        logger.info(
            f"✅ Muestra de referencia validada: Duración={ref_sample.duration:.2f}s "
            f"| Frecuencia={ref_sample.sample_rate}Hz | Transcripción={len(transcript)} chars"
        )
    except AudioValidationError as e:
        logger.error(f"❌ Error de validación de audio de referencia: {e}")
        sys.exit(1)

    # 2. Inicializar motor VoxCPM2
    config = VoxCPM2Config(
        model_id=args.model_id,
        sample_rate=48000,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )
    engine = VoxCPM2Engine(config)
    engine.load_model()

    # 3. Segmentar guion
    bloques = segment_script(guion_raw)
    logger.info(f"📝 Guion segmentado en {len(bloques)} bloques de texto.")

    os.makedirs(DEFAULT_CONFIG.temp_dir, exist_ok=True)
    audios_bloques = []
    total_latency = 0.0

    for i, bloque in enumerate(bloques, start=1):
        logger.info(f"⏳ Procesando bloque {i}/{len(bloques)} ({len(bloque)} caracteres)...")
        temp_chunk_path = os.path.join(DEFAULT_CONFIG.temp_dir, f"bloque_{i:02d}.wav")

        res = engine.synthesize(
            text=bloque,
            reference_audio=ref_sample,
            apply_rioplatense=not args.no_rioplatense,
        )
        save_audio_pcm(temp_chunk_path, res.audio, sample_rate=48000)
        audios_bloques.append(res.audio)
        total_latency += res.latency_seconds

    # 4. Concatenación final a 48 kHz con pausas naturales
    silencio = np.zeros(int(48000 * 0.4), dtype=np.float32)
    piezas = [np.zeros(int(48000 * 0.5), dtype=np.float32)]  # Pausa inicial
    for a in audios_bloques:
        piezas.append(a)
        piezas.append(silencio)

    audio_final = np.concatenate(piezas)
    total_duration = len(audio_final) / 48000
    overall_rtf = total_latency / total_duration if total_duration > 0 else 0.0

    save_audio_pcm(args.output, audio_final, sample_rate=48000, bit_depth="PCM_16")

    # Limpiar carpeta temporal
    import shutil
    shutil.rmtree(DEFAULT_CONFIG.temp_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(f"🎉 ¡Podcast generado con éxito!")
    logger.info(f"📁 Archivo de salida: {args.output}")
    logger.info(f"📊 Frecuencia de muestreo: 48000 Hz (PCM 16-bit)")
    logger.info(f"⏱️ Duración total: {total_duration:.2f}s | Latencia total: {total_latency:.2f}s")
    logger.info(f"⚡ RTF global: {overall_rtf:.4f} (Objetivo: <= {DEFAULT_CONFIG.target_rtf})")
    logger.info("=" * 60)


def cmd_benchmark(args):
    """Ejecuta benchmark de inferencia para medir el RTF con bfloat16 y FlashAttention-2."""
    logger.info("⚡ Ejecutando benchmark de inferencia VoxCPM2 (2B)...")

    # Crear muestra sintética de 12 segundos a 48kHz
    sr = 48000
    duration = 12.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    synthetic_ref_audio = 0.5 * np.sin(2 * np.pi * 300 * t)

    ref_sample = AudioSample(
        audio=synthetic_ref_audio,
        sample_rate=sr,
        duration=duration,
        transcript="Esta es una muestra sintética para pruebas de benchmark de síntesis neural.",
    )

    test_sentences = [
        "Bienvenidos al podcast sobre administración moderna de sistemas y contenedores con Podman.",
        "La arquitectura autorregresiva difusiva de VoxCPM2 permite reproducir la fonética rioplatense con total fidelidad.",
        "El procesamiento a cuarenta y ocho kilo hertz entrega un audio con calidad de estudio broadcast.",
    ]

    engine = VoxCPM2Engine(VoxCPM2Config(dtype=args.dtype, attn_implementation=args.attn))
    engine.load_model()

    rtf_list = []
    for s in test_sentences:
        res = engine.synthesize(text=s, reference_audio=ref_sample)
        rtf_list.append(res.rtf)
        logger.info(f"Texto: '{s[:40]}...' -> RTF: {res.rtf:.4f} (Latencia: {res.latency_seconds:.3f}s)")

    avg_rtf = float(np.mean(rtf_list))
    logger.info("-" * 50)
    logger.info(f"📊 Promedio RTF en Benchmark: {avg_rtf:.4f}")
    if avg_rtf <= 0.15:
        logger.info(f"✅ Criterio de aceptación superado: RTF <= 0.15")
    else:
        logger.warning(f"⚠️ RTF superior al objetivo recomendado ({avg_rtf:.4f} > 0.15)")


def cmd_serve(args):
    """Inicia el servidor API FastAPI con Uvicorn."""
    import uvicorn
    logger.info(f"🌐 Iniciando API ClonVoz en {args.host}:{args.port}...")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(
        description="ClonVoz 2.0 - Síntesis de voz con VoxCPM2 (2B) y Nano-vLLM a 48kHz"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Subcomando: generate
    p_gen = subparsers.add_parser("generate", help="Generar podcast desde guion")
    p_gen.add_argument("--guion", default="guion.txt", help="Ruta al archivo de guion")
    p_gen.add_argument("--audio-ref", default="sampleCorto.wav", help="Audio de muestra")
    p_gen.add_argument("--transcript-ref", default="sampleCorto.txt", help="Transcripción de muestra")
    p_gen.add_argument("--output", default="podcast_completo.wav", help="Archivo WAV de salida")
    p_gen.add_argument("--auto-trim", action="store_true", default=True, help="Recortar muestra a 15s si excede (por defecto: True)")
    p_gen.add_argument("--no-auto-trim", dest="auto_trim", action="store_false", help="Desactivar recorte automático")
    p_gen.add_argument("--model-id", default=DEFAULT_CONFIG.model_id, help="ID o ruta del modelo VoxCPM2")
    p_gen.add_argument("--dtype", default="bfloat16", help="Tipo de datos (bfloat16, float16, float32)")
    p_gen.add_argument("--attn", default="flash_attention_2", help="Implementación de atención")
    p_gen.add_argument("--no-rioplatense", action="store_true", help="Desactivar transformación rioplatense")
    p_gen.set_defaults(func=cmd_generate, auto_trim=True)

    # Subcomando: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Ejecutar benchmark de RTF")
    p_bench.add_argument("--dtype", default="bfloat16", help="Tipo de datos")
    p_bench.add_argument("--attn", default="flash_attention_2", help="Implementación de atención")
    p_bench.set_defaults(func=cmd_benchmark)

    # Subcomando: serve
    p_serve = subparsers.add_parser("serve", help="Levantar servidor FastAPI")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host de escucha")
    p_serve.add_argument("--port", type=int, default=8000, help="Puerto de escucha")
    p_serve.add_argument("--reload", action="store_true", help="Recarga en desarrollo")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if not args.command:
        # Por defecto ejecutar 'generate' si no se especifica subcomando
        args = parser.parse_args(["generate"] + sys.argv[1:])

    args.func(args)


if __name__ == "__main__":
    main()
