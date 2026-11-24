"""Q2 - A loja muda o preco?

Pergunta: para o mesmo produto, no mesmo mes, quanto a escolha da loja muda o preco pago (em %)?

Alvo: preco_relativo = PPK da cotacao / mediana do PPK do mesmo produto no mesmo mes.
      1.10 = 10% acima do preco tipico daquele mes. Assim inflacao e sazonalidade (assunto da Q1)
      somem e todos os produtos ficam na mesma escala.
Atributos: loja (o que queremos medir) + produto, marca e tamanho da embalagem (controles,
      para nao culpar a loja por vender marca cara ou embalagem pequena).
Divisao: treino = primeiros 36 meses, teste = ultimos 12 (o teste nunca e visto no treino).
Modelos: baseline (sempre 1.0), Regressao Linear (da o % de cada loja) e KNN.
O k do KNN e escolhido com 5-fold dentro do treino.

Uso: python scripts/q2_preco_relativo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.data import RAIZ, carregar_limpo  # noqa: E402

FIG = RAIZ / "reports" / "figures"
TAB = RAIZ / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MESES_TESTE = 12
TOP_MARCAS = 5  # por produto; demais viram "outra"
SEED = 42

CATEGORICAS = ["loja", "produto", "marca_grp"]
NUMERICAS = ["quantidade"]


# ---------------------------------------------------------------- dados
def montar_base(remover_outliers: bool) -> pd.DataFrame:
    df = carregar_limpo(remover_outliers=remover_outliers).copy()
    med = df.groupby(["produto", "mes"])["ppk"].transform("median")
    df["preco_relativo"] = df["ppk"] / med

    # Marca: as TOP_MARCAS mais frequentes de cada produto; o resto vira "outra";
    # nula vira "sem marca" (hortifruti, pao e carne quase nunca tem marca).
    marca = df["marca"].fillna("sem marca").str.strip().str.upper()
    top = (
        pd.DataFrame({"produto": df["produto"], "marca": marca})
        .value_counts()
        .groupby(level="produto")
        .head(TOP_MARCAS)
        .reset_index()[["produto", "marca"]]
    )
    chave_top = set(zip(top["produto"], top["marca"]))
    eh_top = np.array([(p, m) in chave_top for p, m in zip(df["produto"], marca)])
    df["marca_grp"] = np.where(eh_top | (marca == "SEM MARCA"), df["produto"] + " | " + marca,
                               df["produto"] + " | outra")
    return df


def dividir(df: pd.DataFrame):
    meses = np.sort(df["mes"].unique())
    corte = pd.Timestamp(meses[-MESES_TESTE])
    treino, teste = df[df["mes"] < corte], df[df["mes"] >= corte]
    assert treino["mes"].max() < teste["mes"].min()
    return treino, teste, corte


def preprocessador(categoricas, numericas, drop_first: bool = False) -> ColumnTransformer:
    # drop="first" so na regressao linear: sem ele as colunas de uma mesma categoria
    # somam 1 e os coeficientes ficam indeterminados (colinearidade perfeita).
    ohe = OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None,
                        sparse_output=False)
    return ColumnTransformer([("cat", ohe, categoricas), ("num", StandardScaler(), numericas)])


# ---------------------------------------------------------------- modelos
def modelos(categoricas, numericas):
    """Cada item: (nome, estimador, grade de parametros).

    Regressao Linear: um numero por loja (interpretavel, responde a questao).
    KNN: preve pela media das k cotacoes mais parecidas; capta casos como
    "loja barata em carne mas cara em cafe", que a linear nao capta."""
    def pipe(est, drop_first=False):
        return Pipeline([("prep", preprocessador(categoricas, numericas, drop_first)), ("m", est)])

    return [
        ("Regressao Linear", pipe(LinearRegression(), drop_first=True), None),
        ("KNN", pipe(KNeighborsRegressor()), {"m__n_neighbors": [10, 30, 100]}),
    ]


def metricas(y, p) -> dict:
    return {
        "MSE": mean_squared_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "MAE": mean_absolute_error(y, p),
        "R2": r2_score(y, p),
    }


def treinar_avaliar(treino, teste, categoricas, numericas, rotulo="", so=None):
    """Treina os modelos (ou so os nomeados em `so`) e mede no teste."""
    X_tr, y_tr = treino[categoricas + numericas], treino["preco_relativo"]
    X_te, y_te = teste[categoricas + numericas], teste["preco_relativo"]
    linhas, ajustados = [], {}

    # Baseline: todo mundo cobra o preco tipico (preco_relativo = 1).
    linhas.append({"modelo": "Baseline (sempre 1.0)", "atributos": rotulo, "params": "",
                   **metricas(y_te, np.ones(len(y_te)))})

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for nome, est, grade in modelos(categoricas, numericas):
        if so and nome not in so:
            continue
        print(f"  treinando {nome} ({rotulo})...", flush=True)
        if grade:
            busca = GridSearchCV(est, grade, cv=cv, scoring="neg_mean_squared_error", n_jobs=2)
            busca.fit(X_tr, y_tr)
            est, params = busca.best_estimator_, busca.best_params_
        else:
            est.fit(X_tr, y_tr)
            params = {}
        ajustados[nome] = est
        linhas.append({"modelo": nome, "atributos": rotulo,
                       "params": ", ".join(f"{k.replace('m__', '')}={v}" for k, v in params.items()),
                       **metricas(y_te, est.predict(X_te))})
    return pd.DataFrame(linhas), ajustados


def efeito_loja(modelo_linear: Pipeline) -> pd.Series:
    """Coeficientes de loja da regressao linear, centrados na loja mediana.

    Com drop='first' a primeira loja vira referencia (coef 0). Somamos ela de volta e
    subtraimos a mediana: 0.05 = cobra 5 pontos percentuais acima da loja mediana."""
    ohe = modelo_linear.named_steps["prep"].named_transformers_["cat"]
    nomes = modelo_linear.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(modelo_linear.named_steps["m"].coef_, index=nomes)
    lojas = ohe.categories_[0]
    ef = pd.Series([0.0] + [coef[f"cat__loja_{l}"] for l in lojas[1:]], index=lojas)
    return (ef - ef.median()).sort_values()


# ---------------------------------------------------------------- execucao
def main():
    df = montar_base(remover_outliers=True)
    treino, teste, corte = dividir(df)
    print(f"cotacoes: {len(df)} | treino: {len(treino)} (ate {treino['mes'].max():%m/%Y}) "
          f"| teste: {len(teste)} (de {corte:%m/%Y})")
    print(f"preco_relativo: mediana={df['preco_relativo'].median():.3f}, "
          f"desvio={df['preco_relativo'].std():.3f}")

    # 1) Comparacao de modelos com todos os atributos.
    print("\n== Modelos (todos os atributos) ==")
    res_todos, ajust = treinar_avaliar(treino, teste, CATEGORICAS, NUMERICAS, "loja+produto+marca+embalagem")
    print(res_todos.round(4).to_string(index=False))

    # 2) Resposta da questao: mesmos modelos SEM a loja. A queda de R2 e o quanto a loja explica.
    print("\n== Sem a loja (so produto+marca+embalagem) ==")
    sem_cat = [c for c in CATEGORICAS if c != "loja"]
    res_sem, _ = treinar_avaliar(treino, teste, sem_cat, NUMERICAS, "produto+marca+embalagem")
    print(res_sem.round(4).to_string(index=False))

    tabela = pd.concat([res_todos, res_sem], ignore_index=True)
    tabela.to_csv(TAB / "q2_metricas.csv", index=False)

    comp = (tabela[tabela["modelo"] != "Baseline (sempre 1.0)"]
            .pivot(index="modelo", columns="atributos", values="R2"))
    comp["ganho_R2_da_loja"] = comp["loja+produto+marca+embalagem"] - comp["produto+marca+embalagem"]
    print("\n== R2 no teste: com vs. sem loja ==")
    print(comp.round(4).to_string())

    # 3) Robustez: regressao linear com os outliers mantidos (decisao #3 de MD/DECISOES.md).
    df_out = montar_base(remover_outliers=False)
    tr_o, te_o, _ = dividir(df_out)
    res_out, _ = treinar_avaliar(tr_o, te_o, CATEGORICAS, NUMERICAS, "com outliers",
                                 so=["Regressao Linear"])
    print("\n== Com outliers (robustez) ==")
    print(res_out[res_out["modelo"].isin(["Baseline (sempre 1.0)", "Regressao Linear"])]
          .round(4).to_string(index=False))

    # 4) Efeito de cada loja (regressao linear: modelo interpretavel).
    ef = efeito_loja(ajust["Regressao Linear"])
    ef.rename("efeito_vs_loja_mediana").to_csv(TAB / "q2_efeito_loja.csv")
    print("\n== Efeito da loja (pontos % vs. loja mediana, mesmo produto/mes/marca/embalagem) ==")
    print((ef * 100).round(1).to_string())

    # Cesta: 1 unidade (kg/L) de cada produto ao preco mediano do ultimo mes, ajustado pela loja.
    ultimo = df["mes"].max()
    cesta = df[df["mes"] == ultimo].groupby("produto")["ppk"].median().sum()
    print(f"\nCesta (1 kg/L de cada um dos {df['produto'].nunique()} produtos, mediana de "
          f"{ultimo:%m/%Y}): R$ {cesta:.2f}")
    print(f"  na loja mais barata ({ef.index[0]}): R$ {cesta * (1 + ef.iloc[0]):.2f}")
    print(f"  na loja mais cara   ({ef.index[-1]}): R$ {cesta * (1 + ef.iloc[-1]):.2f}")
    print(f"  diferenca: R$ {cesta * (ef.iloc[-1] - ef.iloc[0]):.2f} por cesta")

    # ------------------------------------------------------------ figuras
    fig, ax = plt.subplots(figsize=(8, 7))
    cores = ["tab:green" if v < 0 else "tab:red" for v in ef]
    ax.barh(ef.index, ef * 100, color=cores)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("pontos % acima/abaixo da loja mediana")
    ax.set_title("Q2 - Efeito da loja no preço (regressão linear)\nmesmo produto, mês, marca e embalagem")
    fig.tight_layout()
    fig.savefig(FIG / "q2_efeito_loja.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4))
    comp[["produto+marca+embalagem", "loja+produto+marca+embalagem"]].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("R² no teste")
    ax.set_title("Q2 - Quanto a loja acrescenta à explicação do preço")
    ax.legend(["sem loja", "com loja"])
    fig.tight_layout()
    fig.savefig(FIG / "q2_r2_com_sem_loja.png", dpi=130)

    melhor = res_todos.iloc[1:].sort_values("RMSE").iloc[0]["modelo"]
    pred = ajust[melhor].predict(teste[CATEGORICAS + NUMERICAS])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(teste["preco_relativo"], pred, s=3, alpha=0.25)
    lim = [0.4, 2.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("preço relativo real")
    ax.set_ylabel("preço relativo previsto")
    ax.set_title(f"Q2 - Previsto vs. real no teste ({melhor})")
    fig.tight_layout()
    fig.savefig(FIG / "q2_previsto_vs_real.png", dpi=130)

    print("\nsalvo: reports/tables/q2_*.csv e reports/figures/q2_*.png")


if __name__ == "__main__":
    main()
