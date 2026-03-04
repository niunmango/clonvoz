#!/bin/bash

# Script de configuración para ClonVoz
# Ejecutar: bash setup.sh

echo "🎙️ Configurando ClonVoz..."

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

echo "✅ ¡Configuración completada!"
echo ""
echo "Para activar el ambiente virtual, ejecuta:"
echo "  source venv/bin/activate"
echo ""
echo "Para ejecutar el script, usa:"
echo "  python generar_podcast.py"
