"""
Pipeline de validación, carga y normalización estandarizada de audio a 48 kHz.
"""

from dataclasses import dataclass
import math
import os
from typing import Optional, Union
import numpy as np
import soundfile as sf
from scipy import signal


class AudioValidationError(ValueError):
    """Excepción para errores de validación de audio de referencia o entrada."""
    pass


@dataclass
class AudioSample:
    """Estructura estandarizada para muestras de audio de referencia a 48 kHz."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    transcript: str
    path: Optional[str] = None

    def __post_init__(self):
        if self.sample_rate != 48000:
            raise AudioValidationError(
                f"La frecuencia de muestreo debe ser exactamente 48000 Hz, recibido: {self.sample_rate}"
            )
        if not isinstance(self.audio, np.ndarray):
            self.audio = np.asarray(self.audio, dtype=np.float32)
        else:
            self.audio = self.audio.astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 48000) -> np.ndarray:
    """
    Re-muestrea audio a target_sr (por defecto 48000 Hz) usando resample_poly de alta calidad.
    """
    if orig_sr == target_sr:
        return audio

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = signal.resample_poly(audio, up, down, axis=-1)
    return resampled.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convierte audio multicanal a mono promediando los canales."""
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Si tiene forma (canales, muestras) o (muestras, canales)
        if audio.shape[0] < audio.shape[1] and audio.shape[0] in (2, 4, 6, 8):
            return np.mean(audio, axis=0)
        return np.mean(audio, axis=-1)
    else:
        audio_squeezed = np.squeeze(audio)
        if audio_squeezed.ndim <= 2:
            return to_mono(audio_squeezed)
        raise AudioValidationError(f"Forma de audio no soportada: {audio.shape}")


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normaliza la amplitud pico del audio para evitar saturación."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (target_peak / max_val)
    return audio


def load_reference_audio(
    audio_path: str,
    transcript: str,
    min_duration: float = 10.0,
    max_duration: float = 15.0,
    target_sr: int = 48000,
    auto_trim: bool = True,
    normalize: bool = True,
) -> AudioSample:
    """
    Valida y carga una muestra de audio de referencia con transcripción obligatoria.

    Reglas estrictas:
    1. transcript es obligatorio (no vacío) para evitar arrastre de fonemas iniciales y saturación de KV-cache.
    2. Duración debe estar en el rango [min_duration, max_duration] (10.0s - 15.0s).
       - Si duración < min_duration: Lanza AudioValidationError.
       - Si duración > max_duration y auto_trim=False: Lanza AudioValidationError.
       - Si duración > max_duration y auto_trim=True: Recorta a max_duration.
    3. Normaliza a target_sr (48000 Hz) y canal mono en memoria.
    """
    # 1. Validación estricta de transcripción
    if transcript is None or not str(transcript).strip():
        raise AudioValidationError(
            "Reference transcript is required to avoid phoneme drifting and KV-cache saturation. "
            "El campo 'transcript' no puede ser nulo ni estar vacío."
        )
    clean_transcript = str(transcript).strip()

    # 2. Validación de existencia de archivo
    if not os.path.isfile(audio_path):
        raise AudioValidationError(f"El archivo de audio '{audio_path}' no existe o no es accesible.")

    try:
        data, sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        raise AudioValidationError(f"Error al decodificar archivo de audio '{audio_path}': {e}")

    # Convertir a mono
    mono_data = to_mono(data)

    # Re-muestrear a target_sr (48 kHz)
    audio_48k = resample_audio(mono_data, sr, target_sr)

    # Calcular duración exacta
    duration = len(audio_48k) / target_sr

    # 3. Validación estricta de duración [10.0s, 15.0s]
    if duration < min_duration:
        raise AudioValidationError(
            f"Audio duration {duration:.2f}s is below minimum required {min_duration:.1f}s. "
            f"La muestra de referencia debe tener al menos {min_duration:.1f} segundos."
        )

    if duration > max_duration:
        if not auto_trim:
            raise AudioValidationError(
                f"Audio duration {duration:.2f}s exceeds maximum allowed {max_duration:.1f}s. "
                f"Use auto_trim=True para recortar automáticamente o provea una muestra de 10-15s."
            )
        # Recortar a max_duration
        max_samples = int(max_duration * target_sr)
        audio_48k = audio_48k[:max_samples]
        duration = max_duration

    # Normalizar amplitud
    if normalize:
        audio_48k = normalize_peak(audio_48k, target_peak=0.95)

    return AudioSample(
        audio=audio_48k,
        sample_rate=target_sr,
        duration=duration,
        transcript=clean_transcript,
        path=os.path.abspath(audio_path),
    )


def save_audio_pcm(
    file_path: str,
    audio: Union[np.ndarray, list],
    sample_rate: int = 48000,
    bit_depth: str = "PCM_16",
) -> str:
    """
    Guarda un arreglo de audio como archivo WAV PCM a 48 kHz con 16 o 24 bits.
    """
    if sample_rate != 48000:
        raise AudioValidationError(f"Frecuencia de salida requerida: 48000 Hz, recibida: {sample_rate}")

    subtype_map = {
        "PCM_16": "PCM_16",
        "PCM_24": "PCM_24",
        "PCM_32": "PCM_32",
        "FLOAT": "FLOAT",
    }
    subtype = subtype_map.get(bit_depth.upper(), "PCM_16")

    audio_arr = np.asarray(audio, dtype=np.float32)
    audio_arr = to_mono(audio_arr)
    audio_arr = np.clip(audio_arr, -1.0, 1.0)

    # Crear directorio padre si no existe
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    sf.write(file_path, audio_arr, sample_rate, subtype=subtype)
    return file_path
