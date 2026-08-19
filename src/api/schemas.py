"""
Esquemas de validación Pydantic para el pipeline de síntesis de voz VoxCPM2.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class SynthesisRequest(BaseModel):
    """Esquema de solicitud de síntesis individual con validación estricta."""
    text: str = Field(
        ...,
        min_length=1,
        description="Texto que se desea sintetizar con clonación de voz."
    )
    sample_audio_path: str = Field(
        ...,
        min_length=1,
        description="Ruta al archivo WAV de referencia (duración entre 10.0s y 15.0s)."
    )
    sample_transcript: str = Field(
        ...,
        min_length=1,
        description="Transcripción exacta del audio de referencia (obligatoria)."
    )
    auto_trim: bool = Field(
        default=True,
        description="Si es True, recorta automáticamente el audio a 15.0s si excede el rango."
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperatura de muestreo estocástico."
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus sampling) para la generación."
    )
    apply_rioplatense: bool = Field(
        default=True,
        description="Si es True, aplica adaptación fonética rioplatense (ll/y -> sh)."
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Ruta personalizada donde guardar el archivo de salida PCM 48kHz."
    )

    @field_validator("text", "sample_transcript", mode="before")
    @classmethod
    def validate_non_empty_strings(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("El campo no puede estar vacío ni contener solo espacios.")
        return v.strip()


class PodcastRequest(BaseModel):
    """Esquema de solicitud de generación de podcast completo por párrafos."""
    script_text: Optional[str] = Field(
        default=None,
        description="Contenido completo del guion con párrafos separados por líneas en blanco."
    )
    script_path: Optional[str] = Field(
        default=None,
        description="Ruta al archivo guion.txt si no se provee script_text directamente."
    )
    sample_audio_path: str = Field(
        ...,
        description="Ruta al archivo WAV de referencia (10-15s)."
    )
    sample_transcript: str = Field(
        ...,
        min_length=1,
        description="Transcripción exacta de la muestra de referencia."
    )
    auto_trim: bool = Field(
        default=True,
        description="Si es True, recorta la muestra si excede 15s."
    )
    output_path: str = Field(
        default="podcast_completo.wav",
        description="Ruta de destino del audio final normalizado."
    )
    apply_rioplatense: bool = Field(
        default=True,
        description="Habilita adaptación fonética rioplatense."
    )

    @field_validator("sample_transcript", mode="before")
    @classmethod
    def validate_transcript(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("La transcripción de referencia es obligatoria.")
        return v.strip()


class SynthesisResponse(BaseModel):
    """Respuesta de la API tras generar audio a 48 kHz."""
    success: bool = True
    output_audio_path: str
    sample_rate: int = 48000
    duration_seconds: float
    rtf: float
    latency_seconds: float
    metrics: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Estado y configuración del motor de inferencia VoxCPM2."""
    status: str
    model_id: str
    device: str
    sample_rate: int = 48000
    dtype: str
    attn_implementation: str
    target_rtf: float
    inferences_count: int = 0
