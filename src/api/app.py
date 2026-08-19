"""
Servicio REST FastAPI para el motor de síntesis de voz VoxCPM2 y Nano-vLLM.
"""

from contextlib import asynccontextmanager
import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from src.api.schemas import (
    HealthResponse,
    PodcastRequest,
    SynthesisRequest,
    SynthesisResponse,
)
from src.config import DEFAULT_CONFIG
from src.engine.voxcpm2_engine import VoxCPM2Engine
from src.preprocessing.audio_loader import (
    AudioValidationError,
    load_reference_audio,
    save_audio_pcm,
)
from src.preprocessing.text_processor import segment_script

# Instancia global del motor
engine: Optional[VoxCPM2Engine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = VoxCPM2Engine(DEFAULT_CONFIG)
    engine.load_model()
    yield


app = FastAPI(
    title="ClonVoz - VoxCPM2 API",
    description="API de síntesis y clonación de voz a 48 kHz basada en VoxCPM2 (2B) y Nano-vLLM",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Retorna el estado de salud y configuración del motor de inferencia."""
    global engine
    if engine is None:
        engine = VoxCPM2Engine(DEFAULT_CONFIG)
        engine.load_model()

    summary = engine.get_metrics_summary()
    return HealthResponse(
        status="ready" if engine.is_loaded else "initializing",
        model_id=engine.model_id,
        device=engine.device,
        sample_rate=engine.sample_rate,
        dtype=str(engine.torch_dtype) if engine.torch_dtype else "float32",
        attn_implementation=engine.attn_implementation,
        target_rtf=engine.config.target_rtf,
        inferences_count=summary.get("total_inferences", 0),
    )


@app.post("/api/v1/synthesize", response_model=SynthesisResponse)
async def synthesize_text(req: SynthesisRequest):
    """
    Sintetiza un bloque de texto utilizando la muestra de audio de referencia.
    Exige sample_audio_path (10-15s) y sample_transcript exacto.
    """
    global engine
    if engine is None:
        engine = VoxCPM2Engine(DEFAULT_CONFIG)
        engine.load_model()

    try:
        # Cargar y validar audio de referencia (estricto 10-15s y transcripción obligatoria)
        ref_sample = load_reference_audio(
            audio_path=req.sample_audio_path,
            transcript=req.sample_transcript,
            min_duration=DEFAULT_CONFIG.min_reference_duration,
            max_duration=DEFAULT_CONFIG.max_reference_duration,
            target_sr=DEFAULT_CONFIG.sample_rate,
            auto_trim=req.auto_trim,
        )

        # Ejecutar síntesis en VoxCPM2
        result = engine.synthesize(
            text=req.text,
            reference_audio=ref_sample,
            temperature=req.temperature,
            top_p=req.top_p,
            apply_rioplatense=req.apply_rioplatense,
        )

        # Guardar audio de salida a 48 kHz PCM
        out_path = req.output_path or os.path.join(
            DEFAULT_CONFIG.temp_dir, f"synthesis_{int(ref_sample.duration * 1000)}.wav"
        )
        save_audio_pcm(
            file_path=out_path,
            audio=result.audio,
            sample_rate=result.sample_rate,
            bit_depth=DEFAULT_CONFIG.bit_depth,
        )

        return SynthesisResponse(
            success=True,
            output_audio_path=out_path,
            sample_rate=result.sample_rate,
            duration_seconds=result.duration_seconds,
            rtf=result.rtf,
            latency_seconds=result.latency_seconds,
            metrics=result.metrics,
            message="Síntesis completada con éxito a 48 kHz.",
        )

    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error de validación de audio: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la síntesis: {str(e)}",
        )


@app.post("/api/v1/podcast", response_model=SynthesisResponse)
async def synthesize_podcast(req: PodcastRequest):
    """
    Procesa un guion completo párrafo por párrafo y genera el audio unificado.
    """
    global engine
    if engine is None:
        engine = VoxCPM2Engine(DEFAULT_CONFIG)
        engine.load_model()

    # Obtener contenido del guion
    script_text = req.script_text
    if not script_text and req.script_path:
        if not os.path.isfile(req.script_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el archivo de guion '{req.script_path}'",
            )
        with open(req.script_path, "r", encoding="utf-8") as f:
            script_text = f.read()

    if not script_text or not script_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Se requiere 'script_text' o un 'script_path' con contenido válido.",
        )

    try:
        ref_sample = load_reference_audio(
            audio_path=req.sample_audio_path,
            transcript=req.sample_transcript,
            min_duration=DEFAULT_CONFIG.min_reference_duration,
            max_duration=DEFAULT_CONFIG.max_reference_duration,
            target_sr=DEFAULT_CONFIG.sample_rate,
            auto_trim=req.auto_trim,
        )

        bloques = segment_script(script_text)
        if not bloques:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El guion no contiene párrafos válidos.",
            )

        audios_generados = []
        total_latency = 0.0

        for bloque in bloques:
            res = engine.synthesize(
                text=bloque,
                reference_audio=ref_sample,
                apply_rioplatense=req.apply_rioplatense,
            )
            audios_generados.append(res.audio)
            total_latency += res.latency_seconds

        # Concatenar con breve silencio entre bloques (0.3s a 48 kHz)
        import numpy as np
        silencio_inter_bloque = np.zeros(int(DEFAULT_CONFIG.sample_rate * 0.3), dtype=np.float32)
        piezas = []
        for a in audios_generados:
            piezas.append(a)
            piezas.append(silencio_inter_bloque)

        audio_final = np.concatenate(piezas)
        total_duration = len(audio_final) / DEFAULT_CONFIG.sample_rate
        avg_rtf = total_latency / total_duration if total_duration > 0 else 0.0

        save_audio_pcm(
            file_path=req.output_path,
            audio=audio_final,
            sample_rate=DEFAULT_CONFIG.sample_rate,
            bit_depth=DEFAULT_CONFIG.bit_depth,
        )

        return SynthesisResponse(
            success=True,
            output_audio_path=req.output_path,
            sample_rate=DEFAULT_CONFIG.sample_rate,
            duration_seconds=total_duration,
            rtf=avg_rtf,
            latency_seconds=total_latency,
            metrics={
                "bloques_procesados": len(bloques),
                "avg_rtf": round(avg_rtf, 4),
                "total_duration_seconds": round(total_duration, 2),
            },
            message=f"Podcast generado con {len(bloques)} bloques a 48 kHz.",
        )

    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error de validación de audio: {str(e)}",
        )
