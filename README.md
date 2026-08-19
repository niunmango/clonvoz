# ClonVoz - Generador de Podcast con TTS y Clonado de Voz

Generador de podcasts en español con soporte para fonética rioplatense utilizando el modelo neural **Qwen3-TTS** de Hugging Face.

---

## 🚀 Características

- 🎙️ **Síntesis neural y clonación de voz:** Generación de voz natural a partir de un audio y texto de referencia (`sampleCorto.wav` / `sampleCorto.txt`).
- 🇦🇷 **Adaptación Rioplatense:** Reglas de transformación fonética contextual para pronunciación rioplatense (*ll* e *y* a sonido *sh*).
- 📝 **Procesamiento por bloques:** Segmentación automática de guiones por párrafos con reanudación inteligente de bloques ya generados.
- 🎚️ **Mastering y normalización con FFmpeg:** Normalización EBU R128 (-14 LUFS), compresión de rango dinámico y limitador de picos (-1.0 dB).
- ⚡ **Soporte Multiplataforma & Aceleración:** Detección automática de aceleración por hardware (NVIDIA CUDA / AMD ROCm / Apple Metal MPS) con fallback a CPU.
- 🛡️ **Tolerancia a fallos de memoria (OOM):** Detección de errores de memoria en GPU con reintento automático y conmutación transparente a CPU.
- 🔒 **Control de recursos:** Limitación de concurrencia y uso de hilos de CPU para evitar sobrecarga del sistema.

---

## 📋 Requisitos del Sistema

- **Python:** 3.10 o superior.
- **FFmpeg:** Requerido para la concatenación, compresión y normalización final del audio.
  - *Ubuntu/Debian:* `sudo apt-get install ffmpeg`
  - *macOS:* `brew install ffmpeg`
  - *Fedora/RHEL:* `sudo dnf install ffmpeg`
- **GPU (Opcional):** Tarjeta gráfica compatible con CUDA o Apple Silicon para acelerar la inferencia.

---

## 📦 Instalación

### Método Automático

El script de instalación prepara el entorno virtual e instala las dependencias necesarias:

```bash
git clone https://github.com/niunmango/clonvoz.git
cd clonvoz
bash setup.sh
```

### Método Manual

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota sobre PyTorch:** Si necesitas instalar una versión específica de PyTorch para tu plataforma:
> ```bash
> # Para CPU:
> pip install torch soundfile --index-url https://download.pytorch.org/whl/cpu
>
> # Para CUDA 12.x:
> pip install torch soundfile --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 🎙️ Guía de Uso

### 1. Preparar los archivos de entrada

- **`guion.txt`:** Texto del podcast estructurado en párrafos (separados por una línea en blanco). Cada párrafo se procesa como un bloque independiente.
- **`sampleCorto.txt`:** Transcripción exacta del audio de muestra.
- **`sampleCorto.wav`:** Archivo de audio de referencia con la voz que se desea clonar (recomendado: audio limpio, 24kHz, sin ruido de fondo).

### 2. Ejecución

#### En primer plano:
```bash
source venv/bin/activate
python generar_podcast.py
```

#### En segundo plano (con registro de logs):
```bash
./generar.sh
# Para monitorear el progreso en tiempo real:
tail -f salida.log
```

### 3. Salida

- El audio generado final se guarda como **`podcast_completo.wav`** (o con sufijo incremental `podcast_completo_1.wav` si ya existiera el archivo previo).
- La carpeta temporal `temp_audio/` se limpia automáticamente al finalizar la concatenación y normalización.

---

## 📁 Estructura del Proyecto

```
clonvoz/
├── generar_podcast.py     # Script principal de síntesis y procesamiento
├── generar.sh             # Lanzador en segundo plano con control de errores
├── setup.sh               # Asistente de configuración y verificación de dependencias
├── init-git.sh            # Script de inicialización de control de versiones
├── guion.txt              # Guion de entrada del episodio
├── sampleCorto.txt        # Transcripción del audio de referencia
├── sampleCorto.wav        # Audio de muestra para clonación de voz
├── requirements.txt       # Dependencias principales del proyecto
├── .gitignore             # Reglas de exclusión de artefactos generados y logs
└── README.md              # Documentación técnica
```

---

## 🔒 Consideraciones de Seguridad y Buenas Prácticas

1. **Confianza en Modelos:** La carga del modelo utiliza `Qwen/Qwen3-TTS-12Hz-1.7B-Base` desde el repositorio oficial de Hugging Face. Modificar este identificador por repositorios no verificados puede implicar la ejecución de código no seguro mediante `trust_remote_code=True`.
2. **Ejecución de Subprocesos:** Todas las llamadas a utilidades del sistema (`ffmpeg`) se realizan pasando listas de argumentos sin shell intermediaria (`shell=False`) y con límites de tiempo (*timeouts*), mitigando riesgos de inyección de comandos.
3. **Aislamiento de Entorno y Privacidad:**
   - Los archivos de registro (`*.log`, `error_log.txt`) y audios generados están excluidos de Git en `.gitignore` para evitar filtraciones accidentales de rutas o datos procesados.
   - Las variables de entorno (`.env`, `.env.*`) están protegidas y excluidas del control de versiones.

---

## 👤 Autor

- **Ramiro** ([@niunmango](https://github.com/niunmango))

