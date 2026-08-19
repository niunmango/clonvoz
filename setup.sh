#!/usr/bin/env bash
set -euo pipefail

# Script de configuración para ClonVoz 2.0 (VoxCPM2 + Nano-vLLM)
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

# 5. Instalar dependencias
echo "📥 Actualizando pip e instalando dependencias..."
pip install --upgrade pip setuptools wheel

# Configuración de compilación para GPUs NVIDIA RTX serie 4000+ (Ada Lovelace) y CUDA 12.x
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "📥 Instalando paquetes de requirements.txt..."
pip install -r requirements.txt || {
    echo "⚠️ Falló la instalación de algunos binarios opcionales (ej. flash-attn)."
    echo "   Instalando dependencias base en modo CPU/compatibilidad..."
    pip install torch torchaudio soundfile librosa scipy numpy pydantic fastapi uvicorn pytest
}

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
