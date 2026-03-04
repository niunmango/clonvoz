#!/bin/bash

# Script para inicializar git en el proyecto ClonVoz
# Ejecutar: bash init-git.sh

echo "🔧 Inicializando repositorio Git..."

# Inicializar git
git init

# Configurar usuario (opcional, reemplazar con tus datos)
# git config user.email "tu-email@example.com"
# git config user.name "Tu Nombre"

# Hunk size para archivos de audio (si usas git lfs)
# git lfs install
# git lfs track "*.wav"

# Agregar archivos
git add -A

# Crear primer commit
git commit -m "🎙️ Inicial: Proyecto ClonVoz - Generador de podcast con TTS"

echo "✅ Repositorio Git inicializado"
echo ""
echo "Próximos pasos:"
echo "1. Crear un repositorio en GitHub"
echo "2. Ejecutar: git remote add origin <tu-repo-url>"
echo "3. Ejecutar: git branch -M main"
echo "4. Ejecutar: git push -u origin main"
