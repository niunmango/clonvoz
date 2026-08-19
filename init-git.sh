#!/usr/bin/env bash
set -euo pipefail

# Script para inicializar git en el proyecto ClonVoz
# Ejecutar: bash init-git.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "🔧 Inicializando repositorio Git..."

if [ ! -d ".git" ]; then
    git init
    echo "✓ Repositorio Git inicializado"
else
    echo "ℹ️ El repositorio Git ya se encuentra inicializado."
fi

# Agregar archivos
git add .

echo "✅ Archivos preparados en el staging"
echo ""
echo "Próximos pasos recomendados:"
echo "1. git commit -m \"🎙️ Inicial: Proyecto ClonVoz - Generador de podcast con TTS\""
echo "2. Crear un repositorio en GitHub"
echo "3. git remote add origin <tu-repo-url>"
echo "4. git branch -M main"
echo "5. git push -u origin main"

