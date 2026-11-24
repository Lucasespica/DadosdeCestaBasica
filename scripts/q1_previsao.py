"""Q1: previsão de preços 1, 3 e 6 meses à frente (walk-forward), com e sem outliers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import RAIZ, carregar_limpo, serie_produto_mes  # noqa: E402
from src.previsao import metricas, walk_forward  # noqa: E402

FIG = RAIZ / "reports" / "figures"
TAB = RAIZ / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = carregar_limpo()
cenarios = {
    "com outliers": serie_produto_mes(df),
    "sem outliers": serie_produto_mes(df, remover_outliers=True),
}

todas, resumos = [], []
for nome, serie in cenarios.items():
    print(f"\n=== cenário: {nome} ===")
    pred = walk_forward(serie)
    pred["cenario"] = nome
    todas.append(pred)
    m = metricas(pred)
    m["cenario"] = nome
    resumos.append(m)
    print(m.groupby(["modelo", "h"])["mase"].mean().unstack("h").round(3).to_string())

pred = pd.concat(todas, ignore_index=True)
met = pd.concat(resumos, ignore_index=True)
pred.to_csv(TAB / "q1_previsoes.csv", index=False)
met.to_csv(TAB / "q1_metricas_produto.csv", index=False)

# Tabela principal: MASE médio por modelo x horizonte, por cenário.
tab = met.groupby(["cenario", "modelo", "h"])["mase"].mean().unstack("h").round(3)
tab.to_csv(TAB / "q1_mase_modelo_horizonte.csv")

# Em quantos produtos cada modelo vence o naive (MASE menor que o do naive)?
base = met[met["modelo"] == "Naive"].set_index(["cenario", "produto", "h"])["mase"]
met["vence_naive"] = met.apply(lambda r: r["mase"] < base[(r["cenario"], r["produto"], r["h"])], axis=1)
vit = met.groupby(["cenario", "modelo", "h"])["vence_naive"].sum().unstack("h")
vit.to_csv(TAB / "q1_vitorias_sobre_naive.csv")
print("\nProdutos (de 14) em que o modelo bate o naive:")
print(vit.to_string())

# Figura 1: MASE médio por modelo e horizonte (cenário com outliers e sem).
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (nome, sub) in zip(axes, tab.groupby(level="cenario")):
    sub.droplevel("cenario").plot.bar(ax=ax)
    ax.set_title(f"MASE médio ({nome})")
    ax.set_xlabel("")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_ylim(0, 4.6)
    ax.legend(title="horizonte (meses)", loc="upper left", ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q1_mase_modelos.png", dpi=130)

# Figura 2: previsão vs. real (h=3) do melhor modelo e do naive, em 4 produtos.
melhor = tab.loc["com outliers", 3].drop("Naive").idxmin()
print("\nMelhor modelo não naive (h=3, com outliers):", melhor)
sub = pred[(pred["cenario"] == "com outliers") & (pred["h"] == 3) & pred["modelo"].isin(["Naive", melhor])]
prods = ["Café", "Arroz", "Tomate", "Leite"]
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, prod in zip(axes.ravel(), prods):
    s = sub[sub["produto"] == prod]
    real = s[s["modelo"] == "Naive"].set_index("mes_alvo")["real"]
    ax.plot(real.index, real, "k-", label="real")
    for modelo, g in s.groupby("modelo"):
        ax.plot(g["mes_alvo"], g["previsto"], marker="o", ms=3, label=modelo)
    ax.set_title(prod)
    ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Previsão 3 meses à frente nos últimos 12 meses")
fig.tight_layout()
fig.savefig(FIG / "q1_previsao_vs_real.png", dpi=130)
print("tabelas em reports/tables/, figuras em reports/figures/")
