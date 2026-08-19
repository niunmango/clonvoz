#!/usr/bin/env bash
set -euo pipefail

# Obtener directorio del script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d "venv" ]; then
    echo "❌ Entorno virtual 'venv' no encontrado. Ejecuta primero: bash setup.sh"
    exit 1
fi

source venv/bin/activate
nohup python3 generar_podcast.py > salida.log 2>&1 &
PID=$!
echo "🚀 Proceso iniciado en segundo plano (PID: ${PID})"
echo "📋 Puedes seguir los logs con: tail -f salida.log"

