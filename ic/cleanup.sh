#!/bin/bash

# Imagens para serem removidas
files_to_remove=(
  "objective_plot.png"
  "convergence_plot.png"
  "comparison_plot.png"
  "info.txt"
)

# Diretóritos de saídas de treinamento
directories_to_clear=(
  "outputs/loss"
  "outputs/train"
  "outputs/test"
  "outputs/model"
)

# Remoção dos arquivos .png
for file in "${files_to_remove[@]}"; do
  if [ -f "$file" ]; then
    echo "Removendo $file"
    rm "$file"
  else
    echo "$file não encontrado"
  fi
done

# Limpar os diretórios especificados
for dir in "${directories_to_clear[@]}"; do
  if [ -d "$dir" ]; then
    echo "Removendo arquivos em $dir"
    rm -rf "${dir:?}"/*  # ? previne a exclusão em diretórios raiz não intencionais
  else
    echo "$dir não encontrado"
  fi
done

echo "Operação concluída."