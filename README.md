# ClonVoz - Generador de Podcast con TTS Avanzado

Script para generar podcasts en español con pronunciación rioplatense utilizando el modelo Qwen3-TTS de Hugging Face.

## Características

- 🎙️ Generación de audio con síntesis de voz neural
- 🇦🇷 Soporte para acento rioplatense (conversión inteligente de "ll" e "y")
- 📝 Procesamiento de guiones de texto
- 🔊 Referencia de audio personalizable
- ⚙️ Configuración flexible de modelos

## Requisitos

- Python 3.8+
- CUDA (opcional, para aceleración GPU). El script detecta automáticamente la GPU NVIDIA disponible y la usa.

## Instalación

1. **Clonar el repositorio:**
```bash
git clone <tu-repo-url>
cd clonvoz
```

2. **Crear y activar el ambiente virtual:**
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## Uso

1. **Preparar los archivos:**
   - `guion.txt` - Contenido principal del podcast
   - `sampleCorto.txt` - Texto de muestra (para referencia de voz)
   - `sampleCorto.wav` - Audio de referencia para el tono y estilo

2. **Ejecutar el script:**
```bash
python generar_podcast.py
```

- **GPU detección:** el script comprueba varias opciones en este orden:
  1. GPU con CUDA (NVIDIA o AMD/ROCm)
  2. GPU Apple MPS (Mac)
  3. CPU
  
  Muestra el nombre, índice y capacidad de la tarjeta seleccionada. Si no hay
  ninguna disponible, cae a CPU sin interrumpir.

> **OOM & fallback:** durante la generación cada bloque se envuelve en un
> `try/except`. Si el modelo se queda sin memoria en la GPU se vacía la cache,
> se mueve el modelo a CPU y se vuelve a intentar automáticamente. El proceso
> continúa incluso si esto ocurre a mitad del guion.


3. **El script generará:**
   - `podcast_completo.wav` - Audio final del podcast

## Estructura de archivos

```
clonvoz/
├── venv/                    # Ambiente virtual
├── generar_podcast.py       # Script principal
├── guion.txt               # Contenido del podcast
├── sampleCorto.txt         # Texto de referencia
├── sampleCorto.wav         # Audio de referencia
├── requirements.txt        # Dependencias del proyecto
├── .gitignore             # Archivos a ignorar en Git
└── README.md              # Este archivo
```

## Notas técnicas

- El modelo utiliza `Qwen/Qwen3-TTS-12Hz-1.7B-Base` por defecto
- Requiere descargar modelos de Hugging Face (~2GB) en la primera ejecución
- Los warnings de las librerías están suprimidos por defecto

## Troubleshooting

Si encuentra errores al instalar `torch`:

```bash
# Para CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Para CUDA (adaptar la versión según tu CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Licencia

[Especificar tu licencia]

## Autor

Ramiro
