#!/usr/bin/env bash
set -euo pipefail

# Script de configuración para ClonVoz 2.0 (VoxCPM2 + Nano-vLLM)
# Con soporte para GPU NVIDIA y fallback automático a CPU
# Ejecutar: bash setup.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "🎙️ ========================================================"
echo "🎙️ Configurando ClonVoz 2.0 (VoxCPM2 2B + Nano-vLLM)"
echo "🎙️ ========================================================"

# 1. Verificar Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Instálalo para continuar."
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "🐍 Python detectado: v${PY_VER}"

# 2. Verificar ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️ ffmpeg no está instalado en el sistema."
    echo "   Se recomienda instalarlo para el mastering de audio a 48 kHz:"
    echo "   - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "   - macOS: brew install ffmpeg"
    echo "   - Arch Linux: sudo pacman -S ffmpeg"
else
    echo "✓ ffmpeg detectado correctamente."
fi

# 3. Crear venv si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando ambiente virtual en ./venv..."
    python3 -m venv venv
fi

# 4. Activar venv
echo "🔌 Activando ambiente virtual..."
source venv/bin/activate

# 5. Instalar dependencias base
echo "📥 Actualizando pip e instalando herramientas de compilación..."
pip install --upgrade pip setuptools wheel

# 6. Detección de GPU NVIDIA / Fallback a CPU
HAS_NVIDIA=false
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    HAS_NVIDIA=true
    echo "🎮 GPU NVIDIA detectada en el sistema:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    echo "   Configurando flags de compilación para CUDA 12.x / Ada Lovelace / Ampere..."
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
    export MAX_JOBS="${MAX_JOBS:-4}"

    echo "📥 Instalando paquetes con soporte de aceleración GPU..."
    pip install -r requirements.txt || {
        echo "⚠️ Falló la instalación de extensiones C++/CUDA avanzadas (ej. flash-attn)."
        echo "   Instalando dependencias base con soporte de fallback a CPU..."
        pip install torch torchaudio soundfile librosa scipy numpy pydantic fastapi uvicorn pytest
    }
else
    echo "ℹ️ No se detectó GPU NVIDIA disponible. Configurando dependencias en modo CPU..."
    pip install torch torchaudio soundfile librosa scipy numpy pydantic fastapi uvicorn pytest
fi

# Instalar el paquete en modo editable si es posible
pip install -e . --no-deps 2>/dev/null || true

echo ""
echo "✅ ¡Configuración completada con éxito!"
echo ""
echo "Para activar el ambiente virtual:"
echo "  source venv/bin/activate"
echo ""
echo "Para ejecutar la generación de podcast:"
echo "  python generar_podcast.py"
echo ""
echo "Para ejecutar la API REST:"
echo "  python main.py serve --port 8000"
echo ""
echo "Para ejecutar el benchmark de RTF:"
echo "  python main.py benchmark"
