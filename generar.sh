#!/usr/bin/env bash
set -euo pipefail

# ========================================================
# ClonVoz 2.0 - Lanzador de Generación de Podcast (VoxCPM2 2B)
# ========================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Verificar ambiente virtual
if [ ! -d "venv" ]; then
    echo "❌ Entorno virtual 'venv' no encontrado."
    echo "   Por favor ejecuta primero: bash setup.sh"
    exit 1
fi

# 2. Verificar archivos de entrada por defecto
GUION="${GUION:-guion.txt}"
AUDIO_REF="${AUDIO_REF:-sampleCorto.wav}"
TRANSCRIPT_REF="${TRANSCRIPT_REF:-sampleCorto.txt}"
OUTPUT="${OUTPUT:-podcast_completo.wav}"

for f in "$GUION" "$AUDIO_REF" "$TRANSCRIPT_REF"; do
    if [ ! -f "$f" ]; then
        echo "❌ Archivo requerido no encontrado: '$f'"
        exit 1
    fi
done

# 3. Activar entorno virtual y variables de optimización
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 4. Modo primer plano / segundo plano y argumentos extra
FOREGROUND=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --foreground|-f)
            FOREGROUND=true
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

if [ "$FOREGROUND" = true ]; then
    echo "🎙️ Iniciando generación en primer plano (VoxCPM2 @ 48kHz)..."
    python3 main.py generate \
        --guion "$GUION" \
        --audio-ref "$AUDIO_REF" \
        --transcript-ref "$TRANSCRIPT_REF" \
        --output "$OUTPUT" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "🎙️ ========================================================"
    echo "🎙️ Lanzando ClonVoz 2.0 en segundo plano (VoxCPM2 @ 48kHz)"
    echo "🎙️ Guion: $GUION | Muestra: $AUDIO_REF | Salida: $OUTPUT"
    echo "🎙️ ========================================================"

    nohup python3 main.py generate \
        --guion "$GUION" \
        --audio-ref "$AUDIO_REF" \
        --transcript-ref "$TRANSCRIPT_REF" \
        --output "$OUTPUT" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" > salida.log 2>&1 &

    PID=$!
    echo "🚀 Proceso iniciado exitosamente con PID: ${PID}"
    echo "📁 Salida configurada en: ${OUTPUT}"
    echo "📋 Puedes seguir el progreso en tiempo real con:"
    echo "   tail -f salida.log"
    echo ""
    echo "🛑 Para detener el proceso si es necesario:"
    echo "   kill ${PID}"
fi
