"""Q3 - As outras lojas avisam antes?

Pergunta: saber como o preco de um produto variou NAS OUTRAS LOJAS neste mes melhora a previsao
de que ele vai SUBIR NA LOJA no mes seguinte?

Unidade: uma linha por (loja, produto, mes t), a partir do painel loja-produto-mes (mediana).
Rotulo:  sobe = 1 se o PPK da loja em t+1 for mais de LIMIAR_SOBE (2%) maior que em t.
Grupo A ("so a loja"):     variacao da propria loja em t (t vs t-1), produto, mes do ano.
Grupo B ("loja + mercado"): A + variacao mediana das OUTRAS lojas em t para o mesmo produto
                            + diferenca da loja para o preco mediano das outras lojas em t.
Tudo o que entra em X e conhecido no mes t; so o rotulo olha para t+1 (sem vazamento).
Se os modelos acertam mais com B do que com A, o mercado "avisa antes" a loja.

Modelos: 2 baselines, Naive Bayes, Regressao Logistica (pesos interpretaveis) e Random Forest.

Divisao temporal: teste = previsoes cujo mes alvo (t+1) esta nos ultimos 12 meses; treino = antes.
Parametros escolhidos com validacao temporal (TimeSeriesSplit) so dentro do treino.

Uso: python scripts/q3_sobe_proximo_mes.py
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
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.data import RAIZ, carregar_limpo, painel_loja_produto_mes  # noqa: E402

FIG = RAIZ / "reports" / "figures"
TAB = RAIZ / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

LIMIAR_SOBE = 0.02    # "subiu" = alta maior que 2% (abaixo disso e ruido de arredondamento/embalagem)
MIN_OUTRAS_LOJAS = 3  # mercado so e calculado se houver ao menos 3 outras lojas com dado
CLIP_VAR = 0.5        # variacoes extremas (+-50%) sao cortadas para nao dominar a logistica
MESES_TESTE = 12
SEED = 42

CATEGORICAS = ["produto", "mes_do_ano"]
NUM_A = ["var_loja"]
NUM_B = NUM_A + ["var_mercado", "gap_mercado"]


# ---------------------------------------------------------------- dados
def mediana_das_outras(s: pd.Series) -> pd.Series:
    """Para cada linha do grupo, a mediana dos valores das OUTRAS linhas (exclui a propria loja)."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        outros = np.delete(v, i)
        outros = outros[~np.isnan(outros)]
        if len(outros) >= MIN_OUTRAS_LOJAS:
            out[i] = np.median(outros)
    return pd.Series(out, index=s.index)


def montar_base() -> pd.DataFrame:
    painel = painel_loja_produto_mes(carregar_limpo(), remover_outliers=True)

    # Tabela larga (loja, produto) x mes com todos os meses: mes faltante vira NaN, entao
    # "mes anterior" e "mes seguinte" sao sempre o mes de calendario vizinho, nunca um salto.
    meses = pd.date_range(painel["mes"].min(), painel["mes"].max(), freq="MS")
    larga = painel.pivot_table(index=["loja", "produto"], columns="mes", values="ppk").reindex(columns=meses)
    var = larga / larga.shift(1, axis=1) - 1            # t vs t-1 (conhecido em t)
    var_prox = larga.shift(-1, axis=1) / larga - 1      # t+1 vs t (o que queremos prever)

    df = pd.DataFrame({
        "ppk": larga.stack(future_stack=True),
        "var_loja": var.stack(future_stack=True),
        "var_prox": var_prox.stack(future_stack=True),
    }).reset_index().rename(columns={"level_2": "mes"})
    df = df.dropna(subset=["ppk"])

    g = df.groupby(["produto", "mes"])
    df["var_mercado"] = g["var_loja"].transform(mediana_das_outras)
    df["gap_mercado"] = df["ppk"] / g["ppk"].transform(mediana_das_outras) - 1

    df = df.dropna(subset=["var_loja", "var_prox", "var_mercado", "gap_mercado"]).copy()
    for c in ["var_loja", "var_mercado", "gap_mercado"]:
        df[c] = df[c].clip(-CLIP_VAR, CLIP_VAR)
    df["mes_do_ano"] = df["mes"].dt.month
    df["sobe"] = (df["var_prox"] > LIMIAR_SOBE).astype(int)
    return df.sort_values("mes").reset_index(drop=True)


def dividir(df: pd.DataFrame):
    meses = np.sort(df["mes"].unique())
    # O rotulo de t esta em t+1. Teste = alvos nos ultimos 12 meses => t nos ultimos 12 meses com rotulo.
    inicio_teste = pd.Timestamp(meses[-MESES_TESTE])
    treino, teste = df[df["mes"] < inicio_teste], df[df["mes"] >= inicio_teste]
    # Nenhum mes alvo do treino (t+1) cai dentro do teste.
    assert (treino["mes"].max() + pd.DateOffset(months=1)) <= teste["mes"].min()
    return treino, teste


# ---------------------------------------------------------------- modelos
def preprocessador(numericas) -> ColumnTransformer:
    # Padronizar poe os atributos na mesma escala: os pesos da logistica ficam comparaveis.
    return ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def modelos(numericas):
    """(nome, estimador, grade). class_weight='balanced' compensa haver menos 'sobe' que 'nao sobe'
    nos modelos que aceitam pesos; o NB nao aceita (e e um ponto de discussao).

    Naive Bayes: probabilidades supondo atributos independentes (referencia simples).
    Regressao Logistica: perceptron com saida em probabilidade; os pesos mostram o mecanismo.
    Random Forest: ensemble (bagging) de arvores; capta combinacoes nao lineares."""
    def pipe(est):
        return Pipeline([("prep", preprocessador(numericas)), ("m", est)])

    return [
        ("Naive Bayes", pipe(GaussianNB()), None),
        ("Regressao Logistica", pipe(LogisticRegression(class_weight="balanced", max_iter=2000)),
         {"m__C": [0.1, 1, 10]}),
        ("Random Forest", pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                      random_state=SEED, n_jobs=-1)),
         {"m__max_depth": [4, 8], "m__min_samples_leaf": [20, 50]}),
    ]


def metricas(y, p) -> dict:
    return {
        "acuracia": accuracy_score(y, p),
        "precisao": precision_score(y, p, zero_division=0),
        "revocacao": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
    }


def treinar_avaliar(treino, teste, numericas, rotulo):
    cols = numericas + CATEGORICAS
    X_tr, y_tr, X_te, y_te = treino[cols], treino["sobe"], teste[cols], teste["sobe"]
    linhas, ajustados, previsoes = [], {}, {}

    def registrar(nome, pred, params=""):
        previsoes[nome] = pred
        linhas.append({"modelo": nome, "atributos": rotulo, "params": params, **metricas(y_te, pred)})

    registrar("Baseline: nunca sobe", DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te))
    registrar("Baseline: repete o ultimo movimento", (teste["var_loja"] > LIMIAR_SOBE).astype(int).to_numpy())

    cv = TimeSeriesSplit(n_splits=4)  # treino ja esta ordenado por mes: valida sempre no "futuro"
    for nome, est, grade in modelos(numericas):
        print(f"  treinando {nome} ({rotulo})...", flush=True)
        params = ""
        if grade:
            busca = GridSearchCV(est, grade, cv=cv, scoring="f1", n_jobs=2).fit(X_tr, y_tr)
            est = busca.best_estimator_
            params = ", ".join(f"{k.replace('m__', '')}={v}" for k, v in busca.best_params_.items())
        else:
            est.fit(X_tr, y_tr)
        ajustados[nome] = est
        registrar(nome, est.predict(X_te), params)
    return pd.DataFrame(linhas), ajustados, previsoes


# ---------------------------------------------------------------- execucao
def main():
    df = montar_base()
    treino, teste = dividir(df)
    print(f"linhas (loja, produto, mes): {len(df)} | lojas: {df['loja'].nunique()} | "
          f"produtos: {df['produto'].nunique()}")
    print(f"treino: {len(treino)} (t de {treino['mes'].min():%m/%Y} a {treino['mes'].max():%m/%Y}) | "
          f"teste: {len(teste)} (t de {teste['mes'].min():%m/%Y} a {teste['mes'].max():%m/%Y})")
    print(f"proporcao 'sobe' (> {LIMIAR_SOBE:.0%} no mes seguinte): treino {treino['sobe'].mean():.1%}, "
          f"teste {teste['sobe'].mean():.1%}")

    print("\n== Grupo A: so informacao da propria loja ==")
    res_a, aj_a, prev_a = treinar_avaliar(treino, teste, NUM_A, "A: so a loja")
    print(res_a.round(3).to_string(index=False))

    print("\n== Grupo B: loja + outras lojas (mercado) ==")
    res_b, aj_b, prev_b = treinar_avaliar(treino, teste, NUM_B, "B: loja + mercado")
    print(res_b.round(3).to_string(index=False))

    tabela = pd.concat([res_a, res_b], ignore_index=True)
    tabela.to_csv(TAB / "q3_metricas.csv", index=False)

    comp = tabela.pivot(index="modelo", columns="atributos", values="F1")
    comp["ganho_F1_do_mercado"] = comp["B: loja + mercado"] - comp["A: so a loja"]
    print("\n== F1 no teste: A (so a loja) vs. B (loja + mercado) ==")
    print(comp.round(3).sort_values("B: loja + mercado", ascending=False).to_string())

    # Interpretacao: pesos da logistica (atributos padronizados => comparaveis) e importancia do RF.
    log = aj_b["Regressao Logistica"]
    nomes = log.named_steps["prep"].get_feature_names_out()
    pesos = pd.Series(log.named_steps["m"].coef_[0], index=nomes)
    pesos_num = pesos[[f"num__{c}" for c in NUM_B]].rename(lambda n: n.removeprefix("num__"))
    print("\n== Pesos da regressao logistica (B), atributos numericos padronizados ==")
    print("(positivo = aumenta a chance de subir; negativo = diminui)")
    print(pesos_num.round(3).to_string())

    rf = aj_b["Random Forest"]
    imp = pd.Series(rf.named_steps["m"].feature_importances_,
                    index=rf.named_steps["prep"].get_feature_names_out())
    # Soma as colunas one-hot de produto e mes para ficar 1 barra por atributo original.
    imp_agrup = imp.groupby(lambda n: n.split("__")[1].split("_")[0] if n.startswith("cat__")
                            else n.split("__")[1]).sum()
    imp_agrup = imp_agrup.rename({"mes": "mes_do_ano"}).sort_values()
    print("\n== Importancia dos atributos (Random Forest, B) ==")
    print(imp_agrup.round(3).to_string())
    pd.concat({"peso_logistica": pesos_num, "importancia_rf": imp_agrup}, axis=1) \
        .to_csv(TAB / "q3_interpretacao.csv")

    # ------------------------------------------------------------ figuras
    melhor = res_b.sort_values("F1", ascending=False).iloc[0]["modelo"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, prev, titulo in [(axes[0], prev_a, "A: só a loja"), (axes[1], prev_b, "B: loja + mercado")]:
        cm = confusion_matrix(teste["sobe"], prev[melhor])
        ConfusionMatrixDisplay(cm, display_labels=["não sobe", "sobe"]).plot(ax=ax, colorbar=False)
        ax.set_title(titulo)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
    fig.suptitle(f"Q3 - Matriz de confusão no teste ({melhor})")
    fig.tight_layout()
    fig.savefig(FIG / "q3_matriz_confusao.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    comp[["A: so a loja", "B: loja + mercado"]].sort_values("B: loja + mercado") \
        .plot.barh(ax=ax)
    ax.set_xlabel("F1 no teste (classe 'sobe')")
    ax.set_title("Q3 - Informação das outras lojas melhora a previsão?")
    ax.legend(["A: só a loja", "B: loja + mercado"], loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "q3_f1_a_vs_b.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(imp_agrup.index, imp_agrup.values, color="tab:blue")
    ax.set_title("Q3 - Importância dos atributos (Random Forest, B)")
    fig.tight_layout()
    fig.savefig(FIG / "q3_importancia.png", dpi=130)

    print("\nsalvo: reports/tables/q3_*.csv e reports/figures/q3_*.png")


if __name__ == "__main__":
    main()
