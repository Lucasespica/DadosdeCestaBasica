"""Roda a limpeza, mostra o resumo e salva os dados processados em data/processed/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import carregar_bruto, limpar, salvar_processados, serie_produto_mes  # noqa: E402

bruto = carregar_bruto()
print(f"bruto: {len(bruto)} linhas, {bruto['loja'].nunique()} lojas, {bruto['produto'].nunique()} produtos")

limpo = limpar(bruto)
print(f"\nlimpo: {len(limpo)} linhas, {limpo['loja'].nunique()} lojas, {limpo['produto'].nunique()} produtos")

serie = serie_produto_mes(limpo)
print("\nmeses com dado por produto:")
print(serie.notna().sum().sort_values().to_string())

salvar_processados(limpo)
print("\nsalvo em data/processed/")
