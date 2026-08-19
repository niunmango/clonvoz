# ClonVoz 2.0 - Síntesis y Clonación Neural de Voz con VoxCPM2 (2B) y Nano-vLLM

Sistema de generación y clonación de voz neural a **48 kHz** de alta fidelidad basado en el modelo **VoxCPM2 (2B)**, optimizado con **Nano-vLLM**, soporte para tipos de datos `bfloat16`, aceleración con **FlashAttention-2** y adaptación fonética rioplatense.

---

## 🚀 Novedades y Características de la Versión 2.0

- 🧠 **Arquitectura VoxCPM2 (2B):** Síntesis autorregresiva difusiva sin tokenizador para una reproducción fiel de fonemas y prosodia natural.
- 🇦🇷 **Fonética Rioplatense Nativa:** Módulo de transformación fonética contextual con sheísmo (`ll` e `y` a `sh`) y preservación de extranjerismos.
- 🎚️ **Calidad de Estudio a 48 kHz:** Pipeline de procesamiento y exportación en PCM de 16/24 bits a 48000 Hz.
- ⏱️ **Inferencia Ultra-Rápida con Nano-vLLM:** Optimización de KV-cache, `bfloat16` y `FlashAttention-2` con objetivo de rendimiento **RTF <= 0.13 - 0.15** en GPUs NVIDIA RTX serie 4000+ (Ada Lovelace).
- 🛡️ **Pipeline de Datos Estandarizado:** Validación estricta de muestras de referencia con rango obligatorio de **[10.0s, 15.0s]** y transcripción exacta obligatoria (`transcript`) para prevenir saturación de memoria y derivas fonéticas.
- 🌐 **API REST FastAPI & CLI Modular:** Interfaz HTTP moderna para microservicios y CLI (`clonvoz` / `main.py`) para procesamiento por lotes.

---

## 📋 Requisitos del Sistema

- **Python:** 3.10 o superior.
- **FFmpeg:** Requerido para mastering, limitador de picos y normalización EBU R128 (-14 LUFS).
  - *Ubuntu/Debian:* `sudo apt-get install ffmpeg`
  - *macOS:* `brew install ffmpeg`
  - *Arch Linux:* `sudo pacman -S ffmpeg`
- **GPU (Recomendada):** NVIDIA RTX 3000/4000 (Ampere / Ada Lovelace) o superior con CUDA 12.x y `FlashAttention-2`.

---

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/niunmango/clonvoz.git
cd clonvoz
```

### 2. Configuración automática
```bash
bash setup.sh
```

### 3. Configuración manual
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🎙️ Pipeline de Preprocesamiento de Audio (10-15s)

Para garantizar estabilidad en la KV-cache de Nano-vLLM y evitar derivas acústicas, el pipeline impone validaciones estrictas:

1. **Duración:** Las muestras deben durar entre **10.0 y 15.0 segundos**.
   - Muestras `< 10.0s`: Rechazadas inmediatamente con `AudioValidationError`.
   - Muestras `> 15.0s`: Rechazadas en modo estricto o recortadas a 15.0s si se activa `auto_trim=True`.
2. **Transcripción Obligatoria:** Debe proveerse la transcripción exacta del audio de muestra (`sampleCorto.txt` o `sample_transcript`).
3. **Normalización en Memoria:** El audio se remuestrea automáticamente a **48000 Hz mono** en memoria.

---

## 💻 Modos de Uso

### 1. Generación de Podcast vía CLI

```bash
# Activar entorno
source venv/bin/activate

# Ejecución directa
python generar_podcast.py

# O utilizando la CLI avanzada
python main.py generate --guion guion.txt --audio-ref sampleCorto.wav --transcript-ref sampleCorto.txt --output podcast_completo.wav
```

### 2. Ejecución en Segundo Plano

```bash
./generar.sh
# Monitorear logs en tiempo real
tail -f salida.log
```

### 3. Benchmark de Rendimiento (RTF)

Para medir el Factor de Tiempo Real (**Real-Time Factor, RTF**):

```bash
python main.py benchmark --dtype bfloat16 --attn flash_attention_2
```
> **Criterio de Aceptación:** RTF <= 0.15 en GPU RTX 4000 / Ada Lovelace con `bfloat16`.

### 4. Servidor API REST (FastAPI)

```bash
python main.py serve --port 8000
```

Documentación interactiva disponible en `http://localhost:8000/docs`.

#### Endpoints Principales:
- `GET /health`: Estado del motor, dispositivo detectado y configuración.
- `POST /api/v1/synthesize`: Síntesis de texto individual con muestra de referencia validada.
- `POST /api/v1/podcast`: Procesamiento completo de guiones segmentados.

---

## 📁 Estructura del Proyecto

```
clonvoz/
├── pyproject.toml             # Metadatos del proyecto y dependencias PEP 621
├── requirements.txt           # Dependencias de producción y GPU
├── main.py                    # Entrypoint CLI principal (generate, benchmark, serve)
├── generar_podcast.py         # Script retrocompatible de generación de podcast
├── generar.sh                 # Lanzador de procesos en segundo plano
├── setup.sh                   # Script de instalación con soporte CUDA 12.x
├── guion.txt                  # Contenido del podcast (párrafos divididos)
├── sampleCorto.txt            # Transcripción exacta de la muestra
├── sampleCorto.wav            # Audio de referencia para clonación (10-15s)
├── src/
│   ├── config.py              # Parámetros globales (48kHz, RTF target, bfloat16)
│   ├── engine/
│   │   ├── voxcpm2_engine.py  # Runtime VoxCPM2 (2B) con Nano-vLLM y métricas RTF
│   ├── preprocessing/
│   │   ├── audio_loader.py    # Validación estricta 10-15s y resampling a 48kHz
│   │   ├── text_processor.py  # Adaptación fonética rioplatense y segmentador
│   ├── api/
│   │   ├── schemas.py         # Modelos de validación Pydantic V2
│   │   ├── app.py             # Aplicación FastAPI y rutas REST
└── tests/
    ├── test_audio_loader.py   # Tests de validación de duración y 48kHz
    ├── test_text_processor.py # Tests de fonética y segmentación
    ├── test_voxcpm2_engine.py # Tests del motor de inferencia y cálculo de RTF
    ├── test_schemas.py        # Tests de esquemas Pydantic
    └── test_pipeline.py       # Test de integración end-to-end
```

---

## 🧪 Ejecución de Pruebas

```bash
source venv/bin/activate
pytest
```

---

## 👤 Autor

- **Ramiro Garcia** ([@niunmango](https://github.com/niunmango))
