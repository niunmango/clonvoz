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
    echo "❌ Python 3 no está instalado en el sistema. Instálalo para continuar."
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "🐍 Python detectado en el sistema: v${PY_VER}"

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

# 3. Crear o verificar ambiente virtual (./venv)
VENV_DIR="$DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "📦 Creando ambiente virtual en ./venv..."
    python3 -m venv "$VENV_DIR" || {
        echo "❌ Error al crear el ambiente virtual con 'python3 -m venv'."
        echo "   En sistemas Debian/Ubuntu, instala el paquete correspondiente:"
        echo "   sudo apt update && sudo apt install -y python3-venv python3-full"
        exit 1
    }
fi

# 4. Asegurar pip en el entorno virtual
if [ ! -f "$VENV_PIP" ]; then
    echo "🔌 Inicializando pip dentro del ambiente virtual..."
    "$VENV_PYTHON" -m ensurepip --upgrade 2>/dev/null || true
fi

echo "🔌 Ambiente virtual verificado en: $VENV_DIR"

# 5. Instalar dependencias utilizando directamente el Python del venv
# (Evita el error 'externally-managed-environment' de PEP 668 en Debian/Ubuntu)
echo "📥 Actualizando pip e instalando herramientas de compilación..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# 6. Detección de GPU NVIDIA / Configuración de instalación
HAS_NVIDIA=false
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    HAS_NVIDIA=true
    echo "🎮 GPU NVIDIA detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    echo "   Configurando flags de compilación para CUDA 12.x / Ada Lovelace / Ampere..."
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
    export MAX_JOBS="${MAX_JOBS:-4}"

    echo "📥 Instalando paquetes con soporte de aceleración GPU..."
    "$VENV_PYTHON" -m pip install -r requirements.txt || {
        echo "⚠️ Falló la instalación de extensiones C++/CUDA opcionales (ej. flash-attn)."
        echo "   Instalando dependencias base con voxcpm..."
        "$VENV_PYTHON" -m pip install torch torchaudio voxcpm soundfile librosa scipy numpy pydantic fastapi uvicorn tqdm rich pytest pytest-asyncio
    }
else
    echo "ℹ️ No se detectó GPU NVIDIA disponible. Configurando dependencias en modo CPU (VoxCPM 2B)..."
    "$VENV_PYTHON" -m pip install torch torchaudio voxcpm soundfile librosa scipy numpy pydantic fastapi uvicorn tqdm rich pytest pytest-asyncio
fi

# Instalar el paquete en modo editable si es posible
"$VENV_PYTHON" -m pip install -e . --no-deps 2>/dev/null || true

echo ""
echo "✅ ¡Configuración completada con éxito!"
echo ""
echo "Para activar el ambiente virtual manualmente:"
echo "  source venv/bin/activate"
echo ""
echo "Para ejecutar la generación de podcast:"
echo "  ./generar.sh"
echo "  o: python generar_podcast.py"
echo ""
echo "Para ejecutar el benchmark de RTF:"
echo "  python main.py benchmark"
