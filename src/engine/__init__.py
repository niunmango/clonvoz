"""
Módulo de inferencia y runtime para VoxCPM2 y Nano-vLLM.
"""

from src.engine.voxcpm2_engine import (
    SynthesisResult,
    VoxCPM2Engine,
    get_optimal_device,
)

__all__ = [
    "SynthesisResult",
    "VoxCPM2Engine",
    "get_optimal_device",
]
