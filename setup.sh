#!/usr/bin/env bash
set -euo pipefail

# Script de configuración para ClonVoz
# Ejecutar: bash setup.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "🎙️ Configurando ClonVoz..."

# Verificar Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Instálalo para continuar."
    exit 1
fi

# Verificar ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️ ffmpeg no está instalado en el sistema."
    echo "   Se recomienda instalarlo para la normalización de audio:"
    echo "   - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "   - macOS: brew install ffmpeg"
fi

# Crear venv si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando ambiente virtual..."
    python3 -m venv venv
fi

# Activar venv
echo "🔌 Activando ambiente virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ ¡Configuración completada con éxito!"
echo ""
echo "Para activar el ambiente virtual, ejecuta:"
echo "  source venv/bin/activate"
echo ""
echo "Para ejecutar el script, usa:"
echo "  python generar_podcast.py"

