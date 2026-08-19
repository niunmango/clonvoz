"""
Configuración central del sistema ClonVoz para VoxCPM2 y Nano-vLLM.
"""

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class VoxCPM2Config:
    """Configuración del runtime VoxCPM2 y Nano-vLLM."""
    model_id: str = field(
        default_factory=lambda: os.getenv("VOXCPM2_MODEL_ID", "openbmb/VoxCPM2")
    )
    sample_rate: int = 48000
    bit_depth: str = "PCM_16"
    dtype: str = field(
        default_factory=lambda: os.getenv("VOXCPM2_DTYPE", "bfloat16")
    )
    attn_implementation: str = field(
        default_factory=lambda: os.getenv("VOXCPM2_ATTN_IMPL", "flash_attention_2")
    )
    device: Optional[str] = None
    min_reference_duration: float = 10.0
    max_reference_duration: float = 15.0
    target_rtf: float = 0.13
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    max_context_len: int = 4096
    use_nano_vllm: bool = True
    temp_dir: str = "temp_audio"


# Instancia global por defecto
DEFAULT_CONFIG = VoxCPM2Config()
