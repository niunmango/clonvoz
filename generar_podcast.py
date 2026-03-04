import os
import torch
import soundfile as sf
import warnings
import re
import subprocess
import shutil
from qwen_tts import Qwen3TTSModel


def _move_model(model, device: str):
    """Asegura que tanto el wrapper como el módulo subyacente estén en `device`."""
    model.device = torch.device(device)
    try:
        model.model.to(device)
    except Exception:
        pass


# Suprimir TODOS los warnings
warnings.filterwarnings("ignore")
# También silenciar stderr de la librería
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)

# --- CONFIGURACIÓN ---
GUION_FILE = "guion.txt"
SAMPLE_TEXT_FILE = "sampleCorto.txt"
REF_AUDIO_FILE = "sampleCorto.wav"
OUTPUT_FILE = "podcast_completo.wav"

# Modelo a usar (se ha fijado el modelo completo por defecto)
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_NOMBRE = "Completo (1.7B)"

def cargar_texto(archivo):
    with open(archivo, 'r', encoding='utf-8') as f:
        return f.read().strip()

def convertir_a_rioplatense(texto):
    """Convierte pronunciación a acento rioplatense (ll->sh, y->sh inteligentemente)"""
    # 1. Reemplazar "ll" por "sh" (siempre es seguro)
    texto = re.sub(r'll', 'sh', texto, flags=re.IGNORECASE)
    
    texto = re.sub(r'(?<![tTyouU])(?<![yY])yl', 'sh', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<![tTyouU])(?<![yY])y(?=[aeiouáéíóú])', 'sh', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<![tT])oo(?=[aeiouáéíóú])', 'o', texto, flags=re.IGNORECASE)
    
    return texto

def obtener_nombre_archivo_unico(archivo_base):
    """Genera un nombre único si el archivo ya existe (podcast.wav -> podcast_1.wav -> podcast_2.wav)"""
    if not os.path.exists(archivo_base):
        return archivo_base
    
    nombre, ext = os.path.splitext(archivo_base)
    contador = 1
    while os.path.exists(f"{nombre}_{contador}{ext}"):
        contador += 1
    return f"{nombre}_{contador}{ext}"

def normalizar_audio_ffmpeg(input_file, output_file):
    """
    Normaliza el audio usando ffmpeg con filtros de loudness, compresor y limitador.
    1. Loudness Normalization a -14.0 LUFS (EBU R128 standard).
    2. Compressor para reducir el rango dinámico.
    3. Limiter para prevenir picos por encima de -1.0dB.
    """
    print(f"\n🎚️ Normalizando audio con ffmpeg...")

    if not shutil.which('ffmpeg'):
        print("   ⚠️ ffmpeg no encontrado. Asegúrate de tener ffmpeg instalado y en tu PATH.")
        print("      En Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("      En macOS: brew install ffmpeg")
        print("      Saltando normalización de audio.")
        return False

    # Usar un archivo temporal para no sobreescribir el original en caso de error
    temp_normalized = f"{os.path.splitext(input_file)[0]}.tmp.wav"

    # Filtro combinado: compresor y normalizador de volumen.
    # acompressor: comprime el audio para un sonido más consistente.
    # loudnorm: ajusta el volumen general al estándar de -14 LUFS. TP=-1.0 actúa como un limitador de picos.
    filter_chain = "acompressor=threshold=-18dB:ratio=3:1:attack=20:release=250,loudnorm=I=-14:LRA=11:TP=-1.0"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_chain,
        '-ar', '24000',  # Asegurar que la salida mantenga el sample rate
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
            print(f"   ⚠️ La cadena de filtros completa falló. Reintentando solo con 'loudnorm'...")
            print(f"   Error original: {stderr}")

            cmd_fallback = [
                'ffmpeg', '-i', input_file,
                '-af', 'loudnorm=I=-14:LRA=11:TP=-1.0',
                '-ar', '24000',
                '-y', temp_normalized
            ]
            fallback_result = subprocess.run(cmd_fallback, capture_output=True, text=True, check=False, timeout=300)

            if fallback_result.returncode == 0:
                print("   ✓ Fallback a 'loudnorm' exitoso.")
                os.replace(temp_normalized, output_file)
                print(f"   ✅ Audio final guardado en {output_file}\n")
                return True
            else:
                print(f"   ❌ El fallback también falló. No se pudo normalizar el audio.")
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

def segmentar_guion(archivo):
    with open(archivo, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().split('\n\n') if p.strip()]

def get_preferred_device():
    """Determina el mejor dispositivo disponible.
    Prioriza:
      1. GPU CUDA (NVIDIA/ROCm)
      2. Apple MPS
      3. CPU
    También imprime detalles para ayudar al usuario.
    """
    # CUDA/ROCm
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        idx = 0
        gpu_name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        vendor = "NVIDIA" if "NVIDIA" in gpu_name.upper() else "AMD/ROCm"
        print(f"🎮 GPU detectada: {gpu_name} (índice {idx} de {num_gpus}, vendor: {vendor})")
        print(f"📊 CUDA Capability: {capability}")
        return f"cuda:{idx}"
    # Apple Metal Performance Shaders (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🎮 GPU Apple MPS detectada")
        return "mps"
    # Fallback CPU
    print("⚠️ Ninguna GPU detectada, usando CPU")
    return "cpu"


def main():
    # --- LIMITAR RECURSOS ---
    # Limitar CPUs a la mitad (mínimo 1)
    import multiprocessing
    num_cpus = max(1, multiprocessing.cpu_count() // 2)
    torch.set_num_threads(num_cpus)
    print(f"🔒 Limitando CPU a {num_cpus} hilos")
    
    # Determinar dispositivo preferido y limitar memoria si es CUDA
    device = get_preferred_device()
    if device.startswith("cuda"):
        try:
            torch.cuda.set_per_process_memory_fraction(0.8, int(device.split(":")[1]))
            print(f"🔒 Limitando GPU memory al 80% ({device})")
        except Exception:
            # Ignorar si no se puede limitar (p.ej. ROCm)
            pass
    
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

    modelo_nombre = MODEL_NOMBRE
    print(f"✅ Modelo a usar: {modelo_nombre}\n")
    
    # Crear carpeta temp_audio si no existe (pero NO la limpiamos para reutilizar audios)
    if not os.path.exists("temp_audio"):
        os.makedirs("temp_audio")
    
    # Obtener nombre único para el archivo de salida
    output_file = obtener_nombre_archivo_unico(OUTPUT_FILE)
    
    # Convertir a pronunciación rioplatense
    print("🗣️ Aplicando acento rioplatense...")
    referencia_texto = convertir_a_rioplatense(referencia_texto)
    bloques = [convertir_a_rioplatense(bloque) for bloque in bloques]

    # (device ya se determinó antes con get_preferred_device)
    print(f"🔧 PyTorch versión: {torch.__version__}")
    # torch.cuda.is_available puede ser True incluso en entornos ROCm
    print(f"💾 CUDA disponible: {torch.cuda.is_available()}")
    print(f"🔢 Hilos de CPU configurados: {torch.get_num_threads()}")

    print(f"\n🚀 Cargando modelo {modelo_nombre}...")
    print(f"   (esto puede tardar varios minutos la primera vez...)")

    # Si hay GPU queremos que el modelo se cargue directamente en ella.
    # `from_pretrained` acepta kwargs de AutoModel, por ejemplo `device_map`.
    load_kwargs = {} if device == "cpu" else {"device_map": "auto"}
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, trust_remote_code=True, **load_kwargs)

    # moverlo al dispositivo escogido usando helper común
    _move_model(model, device)
    print(f"🎙️ Generando {len(bloques)} bloque(s)...\n")

    log_errores = []

    for i, parrafo in enumerate(bloques):
        temp_file = f"temp_audio/bloque_{i+1:02d}.wav"
        if os.path.exists(temp_file):
            print(f"⏭️  Bloque {i+1}/{len(bloques)} ya existe, saltando...")
            continue

        print(f"⏳ Procesando bloque {i+1}/{len(bloques)}...")
        print(f"   Longitud del texto: {len(parrafo)} caracteres")
        try:
            with torch.no_grad():
                resultado = model.generate_voice_clone(
                    ref_audio=REF_AUDIO_FILE,
                    ref_text=referencia_texto,
                    text=parrafo
                )
                print(f"   ✓ Audio generado")
        except RuntimeError as e:
            err_str = str(e).lower()
            # fallback automático si se queda sin memoria en GPU/MPS
            if ("out of memory" in err_str or "memory" in err_str) and device != "cpu":
                print(f"   ⚠️ OOM en {device}, recargando modelo en CPU y reintentando...")
                # liberar cache y recargar modelo en CPU para asegurar que
                # todo se ejecuta correctamente en la nueva plataforma.
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

                device = "cpu"
                # recargar el modelo **completamente** en CPU para evitar
                # problemas con accelerate y dispositivos mixtos
                model = Qwen3TTSModel.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    device_map="cpu"
                )
                # no hace falta moverlo a mano; ya está en CPU
                # reintentar la generación una vez en CPU
                with torch.no_grad():
                    resultado = model.generate_voice_clone(
                        ref_audio=REF_AUDIO_FILE,
                        ref_text=referencia_texto,
                        text=parrafo
                    )
                print(f"   ✓ Audio generado en CPU")
            else:
                raise

            # --- EXTRAER TENSOR DEL RESULTADO ---
            if isinstance(resultado, (list, tuple)):
                audio_segmento = resultado[0]
                # Si el primer elemento también es lista/tupla, extraer el tensor
                if isinstance(audio_segmento, (list, tuple)):
                    audio_segmento = audio_segmento[0]
            else:
                audio_segmento = resultado

            # Convertir a tensor si es necesario y mover a CPU
            if not isinstance(audio_segmento, torch.Tensor):
                audio_segmento = torch.tensor(audio_segmento)
            
            if audio_segmento.device.type != 'cpu':
                audio_segmento = audio_segmento.cpu()
            
            # Convertir a numpy para guardar con soundfile
            audio_np = audio_segmento.numpy()
            
            # Asegurar dimensión correcta (muestras,) o (canales, muestras)
            if audio_np.ndim > 2:
                audio_np = audio_np.squeeze()
            
            sf.write(temp_file, audio_np.T if audio_np.ndim == 2 else audio_np, 24000)
            print(f"   ✓ Guardado correctamente\n")
            del audio_segmento

        except Exception as e:
            print(f"⚠️ Error en bloque {i+1}: {e}")
            log_errores.append(f"Bloque {i+1}: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            continue
    
    # Guardar log de errores si los hubo
    if log_errores:
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write("Errores durante la generación del podcast:\n\n")
            for error in log_errores:
                f.write(f"- {error}\n")
        print(f"⚠️ Se registraron {len(log_errores)} errores en error_log.txt")

    print("🔗 Uniendo archivos...")
    archivos = sorted([f"temp_audio/{f}" for f in os.listdir("temp_audio") if f.endswith(".wav")])
    if archivos:
        lista_audios = []
        for archivo in archivos:
            audio_data, sr = sf.read(archivo)
            lista_audios.append(audio_data)
        
        # Crear silencio al inicio (0.5 segundos a 24000 Hz)
        silencio = torch.zeros(int(24000 * 0.5))
        
        # Convertir audios y agregar al inicio
        audios_tensor = []
        audios_tensor.append(silencio)  # Silencio inicial
        
        for audio_data in lista_audios:
            audio_tensor = torch.tensor(audio_data)
            # Asegurar que sea 1D o 2D correctamente
            if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.ndim == 1:
                pass  # Ya está correcto
            else:
                audio_tensor = audio_tensor.squeeze()
            audios_tensor.append(audio_tensor)
        
        audio_final = torch.cat(audios_tensor, dim=-1)
        audio_final_np = audio_final.numpy()
        
        # Normalizar volumen (aumentar amplitud sin saturar)
        max_val = torch.max(torch.abs(torch.tensor(audio_final_np)))
        if max_val > 0:
            audio_final_np = audio_final_np / float(max_val) * 0.95
        
        sf.write(output_file, audio_final_np, 24000)
        print(f"✅ ¡Podcast creado! Archivo temporal: {output_file}")
        
        # Aplicar normalización con ffmpeg
        normalizar_audio_ffmpeg(output_file, output_file)
        
        print(f"✅ ¡Hecho! Podcast final guardado en: {output_file}")
        
        # Limpiar la carpeta temp_audio después del éxito
        print("🧹 Limpiando archivos temporales...")
        shutil.rmtree("temp_audio")
        print("✓ Carpeta temp_audio eliminada")

if __name__ == "__main__":
    main()
