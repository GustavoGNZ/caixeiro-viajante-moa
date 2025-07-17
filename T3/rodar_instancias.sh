#!/bin/bash

# Caminhos fixos
SEED=42
CONFIG="config.conf"
NUM_GEN=3
PASTA_INSTANCIAS="instances"
PASTA_SAIDAS="resultados"

# Cria pasta de resultados, se não existir
mkdir -p "$PASTA_SAIDAS"

# Loop pelas 5 instâncias
for arquivo in "$PASTA_INSTANCIAS"/*.dat; do
    nome_base=$(basename "$arquivo" .dat)
    echo "=== Rodando $nome_base ==="
    python3 brkga.py "$SEED" "$CONFIG" "$NUM_GEN" "$arquivo" > "$PASTA_SAIDAS/${nome_base}_log.txt"
done

echo "Concluído! Resultados salvos em $PASTA_SAIDAS/"

