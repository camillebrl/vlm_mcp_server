#!/bin/bash

# Script de démarrage corrigé pour le serveur VLM MCP

# Forcer PyTorch à libérer la mémoire GPU immédiatement
export PYTORCH_NO_CUDA_MEMORY_CACHING="1"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
export CUDA_LAUNCH_BLOCKING=1

# Ajouter src au PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Variables d'environnement pour MCP
export PYTHONUNBUFFERED="1"

# Log de debug vers fichier pour diagnostic uniquement
DEBUG_LOG="/tmp/vlm_server.log"
exec 2>$DEBUG_LOG
echo "=== Démarrage du serveur VLM ===" > $DEBUG_LOG
echo "Date: $(date)" >> $DEBUG_LOG
echo "PYTHONPATH: $PYTHONPATH" >> $DEBUG_LOG


# Message de démarrage simple sur stderr (pour MCP)
echo "🚀 Démarrage du serveur VLM MCP..." >&2

# Lancer le serveur directement avec Python au lieu de poetry run
cd "$SCRIPT_DIR"
exec poetry run python -m vlm_server.cli --log-level INFO