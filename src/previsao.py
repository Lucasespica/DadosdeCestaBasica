"""Q1: previsão de preços mensais com validação walk-forward.

Convenção: `serie` é um DataFrame (índice = mês, colunas = produtos) com PPK mediano.
Para cada mês-alvo `alvo` (posição na série) e horizonte `h`, os modelos veem apenas
`serie.iloc[:alvo - h + 1]`, isto é, dados até o mês `alvo - h`.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEMENTE = 42
HORIZONTES = (1, 3, 6)
N_TESTE = 12          # últimos 12 meses são o conjunto de teste


# ---------------------------------------------------------------- baselines e séries temporais
def prev_naive(y: np.ndarray, h: int, **_) -> float:
    return y[-1]


def prev_media_movel(y: np.ndarray, h: int, **_) -> float:
    return y[-3:].mean()


def prev_ets(y: np.ndarray, h: int, **_) -> float:
    """Holt com tendência amortecida (sem sazonalidade: 3-4 anos de dados são pouco para estimá-la)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ajuste = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                          initialization_method="estimated").fit()
            return float(ajuste.forecast(h)[-1])
        except Exception:
            return y[-1]


MODELOS_SERIE = {
    "Naive": prev_naive,
    "Media movel (3)": prev_media_movel,
    "ETS amortecido": prev_ets,
}


# ---------------------------------------------------------------- aprendizado supervisionado (pooled)
def _features(logp: pd.DataFrame, t: int, mes_alvo: pd.Timestamp) -> pd.DataFrame:
    """Atributos de cada produto no instante t (posição), usando apenas dados até t."""
    atual = logp.iloc[t]
    angulo = 2 * np.pi * mes_alvo.month / 12
    return pd.DataFrame({
        "r1": atual - logp.iloc[t - 1],
        "r2": logp.iloc[t - 1] - logp.iloc[t - 2],
        "r3": atual - logp.iloc[t - 3],
        "r6": atual - logp.iloc[t - 6],
        "dev6": atual - logp.iloc[t - 5:t + 1].mean(),
        "sen_mes": np.sin(angulo),
        "cos_mes": np.cos(angulo),
    })


def _amostras_treino(logp: pd.DataFrame, fim: int, h: int):
    """Pares (atributos em t, retorno log p[t+h]-p[t]) com t+h <= fim-1, de todos os produtos."""
    X, y = [], []
    for t in range(6, fim - h):
        X.append(_features(logp, t, logp.index[t + h]))
        y.append(logp.iloc[t + h] - logp.iloc[t])
    return pd.concat(X), pd.concat(y)


def _modelos_ml():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                               random_state=SEMENTE, n_jobs=1),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000, random_state=SEMENTE)),
    }


# ---------------------------------------------------------------- walk-forward
def walk_forward(serie: pd.DataFrame, horizontes=HORIZONTES, n_teste=N_TESTE) -> pd.DataFrame:
    """Retorna previsões em formato longo: (modelo, produto, h, mes_alvo, real, previsto, escala)."""
    logp = np.log(serie)
    n = len(serie)
    linhas = []
    for h in horizontes:
        for alvo in range(n - n_teste, n):
            fim = alvo - h + 1                      # dados disponíveis: serie.iloc[:fim]
            treino = serie.iloc[:fim]
            escala = treino.diff().abs().mean()    # MAE do naive de 1 passo no treino (denominador do MASE)

            # modelos de série temporal: um ajuste por produto
            for produto in serie.columns:
                y = treino[produto].to_numpy()
                for nome, fn in MODELOS_SERIE.items():
                    linhas.append((nome, produto, h, serie.index[alvo], serie[produto].iloc[alvo],
                                   fn(y, h), escala[produto]))

            # modelos supervisionados: um ajuste com todos os produtos
            X, y_ret = _amostras_treino(logp, fim, h)
            X_prev = _features(logp, fim - 1, serie.index[alvo])
            for nome, modelo in _modelos_ml().items():
                modelo.fit(X.to_numpy(), y_ret.to_numpy())
                ret = modelo.predict(X_prev.to_numpy())
                for produto, r in zip(serie.columns, ret):
                    previsto = treino[produto].iloc[-1] * np.exp(r)
                    linhas.append((nome, produto, h, serie.index[alvo], serie[produto].iloc[alvo],
                                   previsto, escala[produto]))
    return pd.DataFrame(linhas, columns=["modelo", "produto", "h", "mes_alvo", "real", "previsto", "escala"])


def metricas(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE, MAPE e MASE por (modelo, produto, h)."""
    p = pred.copy()
    p["erro"] = (p["real"] - p["previsto"]).abs()
    p["mase"] = p["erro"] / p["escala"]
    p["ape"] = p["erro"] / p["real"]
    return (p.groupby(["modelo", "produto", "h"])
            .agg(mae=("erro", "mean"), mape=("ape", "mean"), mase=("mase", "mean"))
            .reset_index())
