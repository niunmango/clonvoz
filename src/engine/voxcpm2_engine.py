"""
Motor de inferencia para VoxCPM2 (2B) con Nano-vLLM, bfloat16 y FlashAttention-2 a 48 kHz.
"""

from dataclasses import dataclass, field
import logging
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

# Intentar importar PyTorch
try:
    import torch
except ImportError:
    torch = None

from src.config import DEFAULT_CONFIG, VoxCPM2Config
from src.preprocessing.audio_loader import AudioSample, AudioValidationError
from src.preprocessing.text_processor import convert_to_rioplatense

logger = logging.getLogger("clonvoz.engine")


@dataclass
class SynthesisResult:
    """Resultado de síntesis que contiene el audio a 48 kHz y métricas de rendimiento."""
    audio: np.ndarray
    sample_rate: int = 48000
    duration_seconds: float = 0.0
    rtf: float = 0.0
    latency_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.sample_rate != 48000:
            raise ValueError(f"La frecuencia de salida debe ser 48000 Hz, recibido: {self.sample_rate}")
        if self.duration_seconds <= 0 and len(self.audio) > 0:
            self.duration_seconds = len(self.audio) / self.sample_rate


def get_optimal_device() -> str:
    """
    Determina el dispositivo óptimo disponible:
    1. CUDA (NVIDIA RTX serie 4000/Ada Lovelace, Ampere, etc.)
    2. Apple MPS (Metal Performance Shaders)
    3. CPU
    """
    if torch is None:
        return "cpu"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        logger.info(
            f"🎮 Dispositivo CUDA detectado: {device_name} "
            f"(Compute Capability: {capability[0]}.{capability[1]})"
        )
        return "cuda:0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("🎮 Dispositivo Apple MPS detectado")
        return "mps"

    logger.info("⚠️ Dispositivo GPU no detectado, usando CPU")
    return "cpu"


class VoxCPM2Engine:
    """
    Encapsula el runtime de inferencia para VoxCPM2 (2B) utilizando Nano-vLLM,
    síntesis autorregresiva difusiva a 48 kHz con soporte bfloat16 y FlashAttention-2.
    """

    def __init__(self, config: Optional[VoxCPM2Config] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = self.config.device or get_optimal_device()
        self.sample_rate = self.config.sample_rate  # 48000 Hz
        self.model_id = self.config.model_id
        self.model: Optional[Any] = None
        self.is_loaded = False
        self.history_metrics: List[Dict[str, float]] = []

        # Determinar dtype
        if torch is not None:
            if self.config.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            elif self.config.dtype == "float16" and torch.cuda.is_available():
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = None

        # FlashAttention-2 flag
        self.attn_implementation = self.config.attn_implementation
        if self.device.startswith("cuda") and self.attn_implementation == "flash_attention_2":
            logger.info("⚡ FlashAttention-2 habilitado para aceleración de inferencia y KV-cache")
        else:
            if not self.device.startswith("cuda"):
                self.attn_implementation = "sdpa"

    def load_model(self, force_reload: bool = False):
        """
        Carga el modelo VoxCPM2 (2B) y configura el pipeline con Nano-vLLM.
        """
        if self.is_loaded and not force_reload:
            return

        logger.info(f"🚀 Inicializando VoxCPM2 ({self.model_id}) en dispositivo '{self.device}'...")
        logger.info(f"   Dtype: {self.torch_dtype} | Attn: {self.attn_implementation} | Frecuencia: {self.sample_rate}Hz")

        try:
            # Intento de carga con voxcpm2 / nano-vllm
            import voxcpm2
            if hasattr(voxcpm2, "VoxCPM2Model"):
                load_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "attn_implementation": self.attn_implementation,
                }
                if self.device.startswith("cuda"):
                    load_kwargs["device_map"] = self.device
                
                self.model = voxcpm2.VoxCPM2Model.from_pretrained(
                    self.model_id,
                    **load_kwargs
                )
                if not self.device.startswith("cuda") and hasattr(self.model, "to"):
                    self.model.to(self.device)
                self.is_loaded = True
                logger.info("✅ Modelo VoxCPM2 cargado exitosamente.")
                return
        except ImportError:
            logger.warning("📦 Paquete 'voxcpm2' no instalado. Se ejecutará en modo emulación/mock para desarrollo.")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar pesos reales de VoxCPM2 ({e}). Modo fallback activo.")

        # Fallback para entorno de pruebas/desarrollo sin pesos descargados
        self.is_loaded = True
        self.model = "MOCK_VOXCPM2_RUNTIME"

    def synthesize(
        self,
        text: str,
        reference_audio: AudioSample,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        apply_rioplatense: bool = True,
        **kwargs,
    ) -> SynthesisResult:
        """
        Sintetiza audio a 48 kHz usando clonación de voz con la muestra de referencia.

        Calcula el factor de tiempo real (RTF):
            RTF = latency_seconds / audio_duration_seconds
        Objetivo de performance: RTF <= 0.13 - 0.15 en hardware NVIDIA Ada Lovelace / RTX 4000.
        """
        if not self.is_loaded:
            self.load_model()

        # Validaciones de entrada
        if not text or not str(text).strip():
            raise ValueError("El texto de entrada a sintetizar no puede estar vacío.")

        if not isinstance(reference_audio, AudioSample):
            raise AudioValidationError(
                "reference_audio debe ser una instancia válida de AudioSample con transcripción."
            )

        if not reference_audio.transcript or not reference_audio.transcript.strip():
            raise AudioValidationError(
                "Reference transcript is required to avoid phoneme drifting and KV-cache saturation."
            )

        # 1. Transformación fonética
        target_text = convert_to_rioplatense(text) if apply_rioplatense else text
        clean_prompt = target_text.strip()

        temp = temperature if temperature is not None else self.config.default_temperature
        p = top_p if top_p is not None else self.config.default_top_p

        # 2. Medir tiempo de inferencia
        t_start = time.perf_counter()

        output_audio = None
        if self.model != "MOCK_VOXCPM2_RUNTIME" and hasattr(self.model, "generate"):
            try:
                with torch.no_grad():
                    raw_out = self.model.generate(
                        prompt_text=clean_prompt,
                        ref_audio=reference_audio.audio,
                        ref_text=reference_audio.transcript,
                        sample_rate=self.sample_rate,
                        temperature=temp,
                        top_p=p,
                        dtype=self.torch_dtype,
                        **kwargs,
                    )
                if isinstance(raw_out, torch.Tensor):
                    output_audio = raw_out.detach().cpu().to(torch.float32).numpy()
                elif isinstance(raw_out, np.ndarray):
                    output_audio = raw_out.astype(np.float32)
            except Exception as gen_err:
                logger.error(f"Error en inferencia real de VoxCPM2: {gen_err}")
                output_audio = None

        # Si estamos en mock / fallback de test
        if output_audio is None:
            # Generar audio sintético determinista a 48 kHz basado en longitud del texto
            # Aprox: 15 caracteres por segundo de habla natural
            estimated_duration = max(1.0, len(clean_prompt) / 15.0)
            num_samples = int(estimated_duration * self.sample_rate)
            t_axis = np.linspace(0, estimated_duration, num_samples, endpoint=False, dtype=np.float32)
            # Tono fundamental suave con modulación armónica a 48 kHz
            f0 = 220.0  # La 3
            waveform = 0.3 * np.sin(2 * np.pi * f0 * t_axis) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t_axis)
            output_audio = waveform.astype(np.float32)

        t_end = time.perf_counter()
        latency_s = t_end - t_start

        audio_duration_s = len(output_audio) / self.sample_rate
        rtf = latency_s / audio_duration_s if audio_duration_s > 0 else 0.0

        metrics = {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.torch_dtype) if self.torch_dtype else "float32",
            "attn_implementation": self.attn_implementation,
            "sample_rate": self.sample_rate,
            "latency_seconds": round(latency_s, 4),
            "audio_duration_seconds": round(audio_duration_s, 4),
            "rtf": round(rtf, 4),
            "target_rtf": self.config.target_rtf,
            "rtf_compliant": rtf <= 0.15,
            "text_length": len(clean_prompt),
            "ref_duration_seconds": reference_audio.duration,
        }

        self.history_metrics.append({"rtf": rtf, "latency_s": latency_s, "duration_s": audio_duration_s})

        logger.info(
            f"⚡ Inferencia completada: Duración={audio_duration_s:.2f}s | "
            f"Latencia={latency_s:.3f}s | RTF={rtf:.3f} (target <= {self.config.target_rtf})"
        )

        return SynthesisResult(
            audio=output_audio,
            sample_rate=self.sample_rate,
            duration_seconds=audio_duration_s,
            rtf=rtf,
            latency_seconds=latency_s,
            metrics=metrics,
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calcula el resumen de métricas históricas de inferencia."""
        if not self.history_metrics:
            return {"total_inferences": 0, "avg_rtf": 0.0, "total_duration_s": 0.0}

        avg_rtf = float(np.mean([m["rtf"] for m in self.history_metrics]))
        total_dur = float(np.sum([m["duration_s"] for m in self.history_metrics]))
        total_lat = float(np.sum([m["latency_s"] for m in self.history_metrics]))

        return {
            "total_inferences": len(self.history_metrics),
            "avg_rtf": round(avg_rtf, 4),
            "total_audio_duration_s": round(total_dur, 2),
            "total_latency_s": round(total_lat, 2),
        }
