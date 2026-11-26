# Preços de cesta básica em Campinas: previsão e comportamento entre mercados

Trabalho 2 de **Sistemas Inteligentes e Aprendizado de Máquina** (PUC-Campinas, Ciência de Dados e Inteligência Artificial, prof. Fernando Soares de Aguiar Neto), atividade de extensão com dados do Observatório PUC-Campinas.

**Grupo:** Antonio Magalhães Roquete Macedo, Gabriel Lippi Angelini, Lucas Espica Rezende, Rafael Martiniano Nogueira Filho.

## Dados

`data/raw/ICB_DADOS_2026.xlsx`: 46.264 cotações de 30 produtos em 51 estabelecimentos (`SUP_01`...), de set/2022 a ago/2026 (48 meses). Colunas: mês, data da coleta, loja, produto, marca, preço, quantidade da embalagem e **PPK** (preço por kg/L, que é o que usamos para comparar preços).

## Questões

| | Questão | Métodos | Autoria |
|---|---|---|---|
| **Q1** | Dá para prever o preço de um produto 1, 3 e 6 meses à frente melhor que repetir o último valor? | Naive, média móvel, ETS amortecido, Random Forest, MLP; validação walk-forward; MASE | Lucas |
| **Q2** | Para o mesmo produto e mês, quanto a escolha da loja muda o preço? | Baseline, Regressão Linear, KNN; R² com e sem a loja | Rafael |
| **Q3** | Saber o que as outras lojas fizeram antecipa uma alta de preço numa loja no mês seguinte? | Baselines, Naive Bayes, Regressão Logística, Random Forest; F1 | Rafael |

Limpeza e análise exploratória (`src/data.py`, `scripts/limpeza.py`, `scripts/eda.py`) são comuns às três questões.

### Resultados em uma linha por questão
- **Q1:** nenhum modelo supera o naive de forma consistente (MASE 1,30 no naive contra 1,29 no ETS a 1 mês; a 3 meses, 1,90 contra 2,00); preços de cesta básica se comportam quase como passeio aleatório em janelas curtas. Os outliers mudam o MASE em ~0,02 sem trocar a ordem dos modelos.
- **Q2:** a loja explica boa parte do preço: o R² sobe de 0,16-0,19 para 0,35-0,40 ao incluí-la. Da loja mais barata à mais cara há ~25 pontos percentuais de diferença (cerca de R$ 69 numa cesta de 1 kg/L de cada um dos 14 produtos).
- **Q3:** a informação das outras lojas melhora o F1 em todos os modelos (melhor: Random Forest, de 0,51 para 0,56). Loja abaixo do preço das outras e mercado em alta tendem a anteceder uma alta na loja (associação temporal, não causalidade).

## Decisões de tratamento dos dados

(Cada decisão é justificada no código, em `src/data.py`.)
- **PPK**, e não o preço da embalagem, para comparar tamanhos diferentes.
- **Dez/2024** estava com o conteúdo duplicado (923 de 1.846 linhas eram cópias exatas); as cópias foram removidas só nesse mês.
- **Outliers** (PPK fora de 1/3x a 3x da mediana do produto no mês; 55 casos, como pão a R$ 0,10/kg e banana a R$ 999) são **marcados, não removidos**, e os modelos são avaliados com e sem eles.
- **Cobertura**: ficam os 14 produtos com cotação em pelo menos 40 dos 48 meses e as 25 lojas com pelo menos 36 meses (premissa nossa, para ter painel quase completo).
- Agregação mensal por **mediana**, que é robusta a erros de digitação.

## Como rodar

Python 3.11+ (testado em 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python scripts/limpeza.py              # resumo da limpeza (gera data/processed/, opcional)
python scripts/eda.py                  # análise exploratória
python scripts/q1_previsao.py          # Q1 (~3 min)
python scripts/q2_preco_relativo.py    # Q2 (~1-2 min)
python scripts/q3_sobe_proximo_mes.py  # Q3 (~1 min)
```

Os scripts leem `data/raw/ICB_DADOS_2026.xlsx` e gravam tabelas em `reports/tables/` e gráficos em `reports/figures/`. Todos os modelos usam semente fixa (42); pequenas diferenças na última casa decimal entre máquinas são normais.

## Estrutura

```
data/raw/ICB_DADOS_2026.xlsx    # dados originais
src/data.py                     # carga e limpeza (compartilhado)
src/previsao.py                 # modelos e validação walk-forward da Q1
scripts/                        # um script por etapa/questão
reports/tables, reports/figures # saídas geradas
requirements.txt
```

## Limitações

- Séries mensais curtas (48 pontos) e 12 meses de teste: diferenças pequenas entre modelos têm pouco peso estatístico.
- Só 25 lojas e 14 produtos; a mediana mensal esconde a variação dentro do mês.
- Os hiperparâmetros da Q1 foram fixados por justificativa (não otimizados); Q2 e Q3 usam validação cruzada para escolhê-los.
- A Q3 mostra associação temporal entre lojas, não causalidade.
- Não foi feita a comparação com o preço do produtor (os dados não estavam disponíveis em base mensal).
