"""
Módulo API y esquemas de validación Pydantic para ClonVoz.
"""

from src.api.schemas import (
    HealthResponse,
    PodcastRequest,
    SynthesisRequest,
    SynthesisResponse,
)

__all__ = [
    "SynthesisRequest",
    "PodcastRequest",
    "SynthesisResponse",
    "HealthResponse",
]
