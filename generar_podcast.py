import gc
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import warnings
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# Suprimir warnings innecesarios
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)

# --- CONFIGURACIÓN ---
GUION_FILE = "guion.txt"
SAMPLE_TEXT_FILE = "sampleCorto.txt"
REF_AUDIO_FILE = "sampleCorto.wav"
OUTPUT_FILE = "podcast_completo.wav"

# Modelo a usar
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_NOMBRE = "Completo (1.7B)"


def _move_model(model, device: str):
    """Asegura que tanto el wrapper como el módulo subyacente estén en `device`."""
    model.device = torch.device(device)
    try:
        model.model.to(device)
    except Exception:
        pass


def cargar_texto(archivo: str) -> str:
    """Carga y retorna el contenido de un archivo de texto en UTF-8."""
    with open(archivo, 'r', encoding='utf-8') as f:
        return f.read().strip()


def convertir_a_rioplatense(texto: str) -> str:
    """Convierte pronunciación a acento rioplatense (ll->sh, y->sh inteligentemente)."""
    # 1. Reemplazar "ll" por "sh"
    texto = re.sub(r'll', 'sh', texto, flags=re.IGNORECASE)

    # 2. Reglas fonéticas para 'y'
    texto = re.sub(r'(?<![tTyouU])(?<![yY])yl', 'sh', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<![tTyouU])(?<![yY])y(?=[aeiouáéíóú])', 'sh', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<![tT])oo(?=[aeiouáéíóú])', 'o', texto, flags=re.IGNORECASE)

    return texto


def obtener_nombre_archivo_unico(archivo_base: str) -> str:
    """Genera un nombre único si el archivo ya existe (podcast.wav -> podcast_1.wav -> podcast_2.wav)."""
    if not os.path.exists(archivo_base):
        return archivo_base

    nombre, ext = os.path.splitext(archivo_base)
    contador = 1
    while os.path.exists(f"{nombre}_{contador}{ext}"):
        contador += 1
    return f"{nombre}_{contador}{ext}"


def normalizar_audio_ffmpeg(input_file: str, output_file: str) -> bool:
    """
    Normaliza el audio usando ffmpeg con filtros de loudness, compresor y limitador.
    1. Loudness Normalization a -14.0 LUFS (EBU R128 standard).
    2. Compressor para reducir el rango dinámico.
    3. Limiter para prevenir picos por encima de -1.0dB.
    """
    print("\n🎚️ Normalizando audio con ffmpeg...")

    if not shutil.which('ffmpeg'):
        print("   ⚠️ ffmpeg no encontrado. Asegúrate de tener ffmpeg instalado y en tu PATH.")
        print("      En Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("      En macOS: brew install ffmpeg")
        print("      Saltando normalización de audio.")
        return False

    temp_normalized = f"{os.path.splitext(input_file)[0]}.tmp.wav"
    filter_chain = "acompressor=threshold=-18dB:ratio=3:1:attack=20:release=250,loudnorm=I=-14:LRA=11:TP=-1.0"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_chain,
        '-ar', '24000',
        '-y',
        temp_normalized
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )

        if result.returncode == 0:
            print("   ✓ Audio normalizado y comprimido correctamente:")
            print("     - Target Loudness: -14.0 LUFS")
            print("     - Compressor: Threshold -18dB, Ratio 3:1")
            print("     - Peak limited a: -1.0dB")
            os.replace(temp_normalized, output_file)
            print(f"   ✅ Audio final guardado en {output_file}\n")
            return True
        else:
            stderr = result.stderr or ''
            print("   ⚠️ La cadena de filtros completa falló. Reintentando solo con 'loudnorm'...")
            print(f"   Error original: {stderr}")

            cmd_fallback = [
                'ffmpeg', '-i', input_file,
                '-af', 'loudnorm=I=-14:LRA=11:TP=-1.0',
                '-ar', '24000',
                '-y', temp_normalized
            ]
            fallback_result = subprocess.run(
                cmd_fallback,
                capture_output=True,
                text=True,
                check=False,
                timeout=300
            )

            if fallback_result.returncode == 0:
                print("   ✓ Fallback a 'loudnorm' exitoso.")
                os.replace(temp_normalized, output_file)
                print(f"   ✅ Audio final guardado en {output_file}\n")
                return True
            else:
                print("   ❌ El fallback también falló. No se pudo normalizar el audio.")
                print(f"   Error del fallback: {fallback_result.stderr}")
                if os.path.exists(temp_normalized):
                    os.remove(temp_normalized)
                return False

    except subprocess.TimeoutExpired:
        print("   ❌ La normalización tomó demasiado tiempo (timeout).")
        if os.path.exists(temp_normalized):
            os.remove(temp_normalized)
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado durante la normalización: {e}")
        if os.path.exists(temp_normalized):
            os.remove(temp_normalized)
        return False


def segmentar_guion(archivo: str) -> list[str]:
    """Lee el archivo y separa los párrafos divididos por líneas en blanco."""
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
    return [p.strip() for p in re.split(r'\n\s*\n', contenido) if p.strip()]


def get_preferred_device() -> str:
    """
    Determina el mejor dispositivo disponible:
      1. GPU CUDA (NVIDIA/ROCm)
      2. Apple MPS
      3. CPU
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num_gpus = torch.cuda.device_count()
        idx = 0
        gpu_name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        vendor = "NVIDIA" if "NVIDIA" in gpu_name.upper() else "AMD/ROCm"
        print(f"🎮 GPU detectada: {gpu_name} (índice {idx} de {num_gpus}, vendor: {vendor})")
        print(f"📊 CUDA Capability: {capability}")
        return f"cuda:{idx}"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🎮 GPU Apple MPS detectada")
        return "mps"

    print("⚠️ Ninguna GPU detectada, usando CPU")
    return "cpu"


def main():
    # --- LIMITAR RECURSOS ---
    num_cpus = max(1, multiprocessing.cpu_count() // 2)
    torch.set_num_threads(num_cpus)
    print(f"🔒 Limitando CPU a {num_cpus} hilos")

    # --- VERIFICACIONES PREVIAS ---
    if not shutil.which('ffmpeg'):
        print("❌ ffmpeg no encontrado. Instala ffmpeg antes de ejecutar este script.")
        print("   En Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("   En macOS: brew install ffmpeg")
        return

    if not shutil.which('sox'):
        print("⚠️ sox no encontrado. Si tu flujo lo requiere instala sox o ignora este aviso.")

    device = get_preferred_device()

    # Validar archivos de entrada
    print("📁 Validando archivos de entrada...")
    archivos_requeridos = [GUION_FILE, SAMPLE_TEXT_FILE, REF_AUDIO_FILE]
    for archivo in archivos_requeridos:
        if not os.path.exists(archivo):
            print(f"   ❌ Error: No se encontró '{archivo}'")
            return
        print(f"   ✓ {archivo} encontrado")

    print("📖 Cargando textos...")
    referencia_texto = cargar_texto(SAMPLE_TEXT_FILE)
    bloques = segmentar_guion(GUION_FILE)

    if not bloques:
        print("❌ El archivo de guion no contiene párrafos válidos.")
        return

    modelo_nombre = MODEL_NOMBRE
    print(f"✅ Modelo a usar: {modelo_nombre}\n")

    # Crear carpeta temp_audio si no existe
    os.makedirs("temp_audio", exist_ok=True)

    # Obtener nombre único para el archivo de salida
    output_file = obtener_nombre_archivo_unico(OUTPUT_FILE)

    # Convertir a pronunciación rioplatense
    print("🗣️ Aplicando acento rioplatense...")
    referencia_texto = convertir_a_rioplatense(referencia_texto)
    bloques = [convertir_a_rioplatense(bloque) for bloque in bloques]

    print(f"🔧 PyTorch versión: {torch.__version__}")
    print(f"💾 CUDA disponible: {torch.cuda.is_available()}")
    print(f"🔢 Hilos de CPU configurados: {torch.get_num_threads()}")

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        print("🗑️ Liberando caché CUDA antes de cargar el modelo")

    print(f"\n🚀 Cargando modelo {modelo_nombre}...")
    print("   (esto puede tardar varios minutos la primera vez...)")

    load_kwargs = {"device_map": "cuda"} if device.startswith("cuda") else {}
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, trust_remote_code=True, **load_kwargs)
    _move_model(model, device)

    print(f"🎙️ Generando {len(bloques)} bloque(s)...\n")

    log_errores = []

    for i, parrafo in enumerate(bloques):
        temp_file = f"temp_audio/bloque_{i+1:02d}.wav"
        if os.path.exists(temp_file):
            print(f"⏭️  Bloque {i+1}/{len(bloques)} ya existe, saltando...")
            continue

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()

        print(f"⏳ Procesando bloque {i+1}/{len(bloques)}...")
        print(f"   Longitud del texto: {len(parrafo)} caracteres")

        resultado = None
        try:
            with torch.no_grad():
                resultado = model.generate_voice_clone(
                    ref_audio=REF_AUDIO_FILE,
                    ref_text=referencia_texto,
                    text=parrafo
                )
            print("   ✓ Audio generado")
        except RuntimeError as e:
            err_str = str(e).lower()
            if ("out of memory" in err_str or "memory" in err_str) and device != "cpu":
                print(f"   ⚠️ OOM en {device}, recargando modelo en CPU y reintentando...")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

                device = "cpu"
                model = Qwen3TTSModel.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    device_map="cpu"
                )
                _move_model(model, "cpu")

                try:
                    with torch.no_grad():
                        resultado = model.generate_voice_clone(
                            ref_audio=REF_AUDIO_FILE,
                            ref_text=referencia_texto,
                            text=parrafo
                        )
                    print("   ✓ Audio generado en CPU")
                except Exception as inner_e:
                    print(f"⚠️ Error en reintento en CPU para bloque {i+1}: {inner_e}")
                    log_errores.append(f"Bloque {i+1}: {str(inner_e)}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    continue
            else:
                print(f"⚠️ Error en bloque {i+1}: {e}")
                log_errores.append(f"Bloque {i+1}: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                continue
        except Exception as e:
            print(f"⚠️ Error en bloque {i+1}: {e}")
            log_errores.append(f"Bloque {i+1}: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            continue

        if resultado is None:
            print(f"⚠️ No se generó audio para el bloque {i+1}")
            continue

        # --- EXTRAER TENSOR DEL RESULTADO ---
        if isinstance(resultado, (list, tuple)):
            audio_segmento = resultado[0]
            if isinstance(audio_segmento, (list, tuple)):
                audio_segmento = audio_segmento[0]
        else:
            audio_segmento = resultado

        if not isinstance(audio_segmento, torch.Tensor):
            audio_segmento = torch.tensor(audio_segmento)
        if audio_segmento.device.type != 'cpu':
            audio_segmento = audio_segmento.cpu()

        audio_np = audio_segmento.numpy()
        if audio_np.ndim > 2:
            audio_np = audio_np.squeeze()

        sf.write(temp_file, audio_np.T if audio_np.ndim == 2 else audio_np, 24000)
        print("   ✓ Guardado correctamente\n")
        del audio_segmento

    # Guardar log de errores si los hubo
    if log_errores:
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write("Errores durante la generación del podcast:\n\n")
            for error in log_errores:
                f.write(f"- {error}\n")
        print(f"⚠️ Se registraron {len(log_errores)} errores en error_log.txt")

    print("🔗 Uniendo archivos...")
    if os.path.exists("temp_audio"):
        archivos = [
            os.path.join("temp_audio", f)
            for f in os.listdir("temp_audio")
            if f.startswith("bloque_") and f.endswith(".wav")
        ]

        def _extraer_numero_bloque(path: str) -> int:
            m = re.search(r"bloque_(\d+)\.wav", os.path.basename(path))
            return int(m.group(1)) if m else 0

        archivos.sort(key=_extraer_numero_bloque)
    else:
        archivos = []

    podcast_success = False
    if archivos:
        lista_audios = []
        for archivo in archivos:
            audio_data, sr = sf.read(archivo)
            lista_audios.append(audio_data)

        # Crear silencio al inicio (0.5 segundos a 24000 Hz)
        silencio = torch.zeros(int(24000 * 0.5))

        audios_tensor = [silencio]
        for audio_data in lista_audios:
            audio_tensor = torch.tensor(audio_data)
            if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.ndim == 1:
                pass
            else:
                audio_tensor = audio_tensor.squeeze()
            audios_tensor.append(audio_tensor)

        audio_final = torch.cat(audios_tensor, dim=-1)
        audio_final_np = audio_final.numpy()

        # Normalizar volumen preliminar
        max_val = torch.max(torch.abs(torch.tensor(audio_final_np)))
        if max_val > 0:
            audio_final_np = audio_final_np / float(max_val) * 0.95

        sf.write(output_file, audio_final_np, 24000)
        print(f"✅ ¡Podcast creado! Archivo temporal: {output_file}")

        # Aplicar normalización con ffmpeg
        normalizar_audio_ffmpeg(output_file, output_file)

        print(f"✅ ¡Hecho! Podcast final guardado en: {output_file}")
        podcast_success = True
    else:
        print("❌ No se encontraron bloques de audio para unir.")

    # Limpiar la carpeta temp_audio sólo si todo salió bien
    if podcast_success and os.path.exists("temp_audio"):
        print("🧹 Limpiando archivos temporales...")
        shutil.rmtree("temp_audio")
        print("✓ Carpeta temp_audio eliminada")


if __name__ == "__main__":
    main()
