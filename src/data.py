"""Carga e limpeza dos dados de cesta básica (ICB_DADOS_2026.xlsx).

Uso:
    from src.data import carregar_limpo, serie_produto_mes, painel_loja_produto_mes
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "ICB_DADOS_2026.xlsx"
PROCESSED = RAIZ / "data" / "processed"

COLUNAS = {
    "Mes_Ano_Data": "mes",
    "Data_Coleta": "data_coleta",
    "Estab": "loja",
    "Produto": "produto",
    "Marca": "marca",
    "Preco": "preco",
    "Quantidade": "quantidade",
    "PPK": "ppk",
}

# Nomes diferentes para o mesmo produto (aparecem em apenas 1 mes cada).
ALIASES = {
    "Carne Acem": "Carne Bovina Acem",
    "Carne Coxão Mole": "Carne Bovina Coxão Mole",
    "Farinha": "Farinha de Trigo",
    "Macarrão": "Macarrão com Ovos",
    "Pão": "Pão Francês",
    "Pernil": "Carne Suína Pernil",
    "Carne Pernil": "Carne Suína Pernil",
}

# Mes cujo conteudo aparece duplicado no arquivo original.
DUPLICADO_MES = pd.Timestamp("2024-12-01")
# Produto fica se tiver cotacoes em pelo menos este numero de meses (de 48).
MIN_MESES_PRODUTO = 40
# Loja fica se tiver cotacoes em pelo menos este numero de meses.
MIN_MESES_LOJA = 36
# Cotacao e outlier se PPK / mediana(produto, mes) fugir deste intervalo.
RAZAO_OUTLIER = (1 / 3, 3.0)


def carregar_bruto() -> pd.DataFrame:
    """Le o xlsx original e renomeia colunas, sem alterar valores."""
    df = pd.read_excel(RAW).rename(columns=COLUNAS)
    df["marca"] = df["marca"].astype("string")
    return df


def limpar(df: pd.DataFrame, remover_outliers: bool = False, log: bool = True) -> pd.DataFrame:
    """Aplica as regras de limpeza, na ordem, e imprime quantas linhas cada uma remove.

    Outliers são apenas marcados por padrão (ver RAZAO_OUTLIER)."""
    def passo(nome, antes, depois):
        if log:
            print(f"[limpeza] {nome}: {antes - depois} linhas removidas ({depois} restantes)")

    df = df.copy()
    df["produto"] = df["produto"].str.strip().replace(ALIASES)
    df["loja"] = df["loja"].str.strip()

    # Dez/2024 foi carregado duas vezes (923 de 1846 linhas são cópias exatas);
    # nos demais meses, linhas idênticas são cotações legítimas e ficam.
    n = len(df)
    copia = df.duplicated(keep="first") & (df["mes"] == DUPLICADO_MES)
    df = df[~copia]
    passo(f"duplicação de {DUPLICADO_MES:%m/%Y}", n, len(df))

    n = len(df)
    df = df[(df["preco"] > 0) & (df["quantidade"] > 0)]
    passo("preço/quantidade inválidos", n, len(df))

    # Outlier relativo à mediana do mesmo produto no mesmo mês (pega R$ 999 da banana).
    # Só marca (coluna `outlier`); quem consome decide se filtra (remover_outliers=True).
    med = df.groupby(["produto", "mes"])["ppk"].transform("median")
    df["outlier"] = ~(df["ppk"] / med).between(*RAZAO_OUTLIER)
    if log:
        print(f"[limpeza] outliers marcados (PPK fora de {RAZAO_OUTLIER[0]:.2f}x-{RAZAO_OUTLIER[1]:.0f}x "
              f"da mediana produto-mês): {int(df['outlier'].sum())}")
    if remover_outliers:
        n = len(df)
        df = df[~df["outlier"]]
        passo("outliers removidos", n, len(df))

    # Produtos com cobertura temporal suficiente.
    meses_prod = df.groupby("produto")["mes"].nunique()
    produtos_ok = meses_prod[meses_prod >= MIN_MESES_PRODUTO].index
    n = len(df)
    df = df[df["produto"].isin(produtos_ok)]
    passo(f"produtos com < {MIN_MESES_PRODUTO} meses", n, len(df))

    # Lojas com cobertura suficiente.
    meses_loja = df.groupby("loja")["mes"].nunique()
    lojas_ok = meses_loja[meses_loja >= MIN_MESES_LOJA].index
    n = len(df)
    df = df[df["loja"].isin(lojas_ok)]
    passo(f"lojas com < {MIN_MESES_LOJA} meses", n, len(df))

    return df.reset_index(drop=True)


def carregar_limpo(remover_outliers: bool = False, log: bool = False) -> pd.DataFrame:
    return limpar(carregar_bruto(), remover_outliers=remover_outliers, log=log)


def painel_loja_produto_mes(df: pd.DataFrame, remover_outliers: bool = False) -> pd.DataFrame:
    """Uma linha por (loja, produto, mês): mediana do PPK entre as cotações do mês."""
    if remover_outliers:
        df = df[~df["outlier"]]
    return (
        df.groupby(["loja", "produto", "mes"], as_index=False)
        .agg(ppk=("ppk", "median"), n_cotacoes=("ppk", "size"))
    )


def serie_produto_mes(df: pd.DataFrame, remover_outliers: bool = False) -> pd.DataFrame:
    """Série nacional: mediana mensal do PPK por produto. Colunas = produtos, índice = mês."""
    if remover_outliers:
        df = df[~df["outlier"]]
    return df.groupby(["mes", "produto"])["ppk"].median().unstack("produto")


def salvar_processados(df: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "cotacoes_limpas.csv", index=False)
    painel_loja_produto_mes(df).to_csv(PROCESSED / "painel_loja_produto_mes.csv", index=False)
    serie_produto_mes(df).to_csv(PROCESSED / "serie_produto_mes.csv")
