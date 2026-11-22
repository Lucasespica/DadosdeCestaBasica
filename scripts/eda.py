"""EDA: outliers marcados, duplicatas, pico de dez/2024 e séries mensais por produto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import RAIZ, carregar_bruto, carregar_limpo, serie_produto_mes  # noqa: E402

FIG = RAIZ / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

bruto = carregar_bruto()
df = carregar_limpo()

print("== Outliers marcados, por produto ==")
out = df[df["outlier"]]
print(out.groupby("produto").size().sort_values(ascending=False).to_string())
print("\nMaiores desvios:")
print(out.sort_values("ppk").iloc[[0, 1, 2, -3, -2, -1]][["mes", "loja", "produto", "preco", "quantidade", "ppk"]].to_string())

print("\n== Duplicatas exatas (bruto) ==")
dup = bruto.duplicated(keep=False)
print("linhas em grupos duplicados:", int(dup.sum()), "| a remover se mantiver 1:", int(bruto.duplicated().sum()))
print("duplicatas por mês (top 5):")
print(bruto[bruto.duplicated()].groupby("mes").size().sort_values(ascending=False).head(5).to_string())

print("\n== Cotações por mês (bruto) e lojas por mês ==")
por_mes = bruto.groupby("mes").agg(cotacoes=("ppk", "size"), lojas=("loja", "nunique"))
print(por_mes.loc["2024-10-01":"2025-02-01"].to_string())
dez = bruto[bruto["mes"] == "2024-12-01"]
print("dez/2024: cotações por loja (média):", round(dez.groupby("loja").size().mean(), 1),
      "| datas de coleta distintas:", dez["data_coleta"].nunique())
print("cotações por loja-produto em dez/2024 vs. nov/2024:",
      round(dez.groupby(["loja", "produto"]).size().mean(), 2),
      round(bruto[bruto["mes"] == "2024-11-01"].groupby(["loja", "produto"]).size().mean(), 2))

print("\n== Cotações por loja-produto-mês (dados limpos) ==")
print(df.groupby(["loja", "produto", "mes"]).size().value_counts().sort_index().to_string())

# Séries mensais nacionais, com e sem outliers marcados.
com = serie_produto_mes(df)
sem = serie_produto_mes(df, remover_outliers=True)
print("\nMaior diferença relativa entre série com e sem outliers (por produto):")
print(((com - sem).abs() / sem).max().sort_values(ascending=False).round(3).to_string())

fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
for ax, prod in zip(axes.ravel(), com.columns):
    ax.plot(com.index, com[prod], lw=1.5, label="com outliers")
    ax.plot(sem.index, sem[prod], lw=1, ls="--", label="sem outliers")
    ax.set_title(prod, fontsize=9)
    ax.tick_params(labelsize=7)
for ax in axes.ravel()[len(com.columns):]:
    ax.axis("off")
axes[0, 0].legend(fontsize=7)
fig.suptitle("PPK mediano mensal por produto (R$/kg ou L)")
fig.tight_layout()
fig.savefig(FIG / "eda_series_produto.png", dpi=130)
print("\nsalvo:", FIG / "eda_series_produto.png")

fig, ax = plt.subplots(figsize=(9, 3.5))
por_mes["cotacoes"].plot(ax=ax, marker="o", ms=3)
ax.set_title("Cotações por mês (bruto)")
fig.tight_layout()
fig.savefig(FIG / "eda_cotacoes_por_mes.png", dpi=130)
