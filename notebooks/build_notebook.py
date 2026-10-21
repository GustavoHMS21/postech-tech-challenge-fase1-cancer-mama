"""Gera notebooks/analysis.ipynb a partir de código Python.

Por que gerar o notebook por script? (1) o .ipynb é JSON difícil de revisar/versionar;
(2) garante que o notebook só CHAMA funções de src/ (sem duplicar lógica);
(3) `python notebooks/build_notebook.py` recria e executa o notebook do zero.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Tech Challenge Fase 1 — Apoio ao diagnóstico de câncer de mama
Protótipo acadêmico: classificar tumores como **maligno (1)** ou **benigno (0)** a partir do dataset público
*Breast Cancer Wisconsin (Diagnostic)*. O modelo é **apoio à decisão** — a palavra final é sempre do médico.

Este notebook apenas chama as funções de `src/` (onde está a lógica testada) e discute os resultados,
seguindo a ordem do enunciado: dados → exploração → pré-processamento → modelos → avaliação → interpretação."""),
code("""# Permite importar src/ a partir da pasta notebooks/
import sys; sys.path.append("..")
import pandas as pd
from IPython.display import Image, display
from src import data, eda, preprocessing, train, evaluate, explain"""),

md("""## 1. Problema e dados
O dataset tem 569 exames (aspirado por agulha fina) com 30 medidas numéricas do núcleo celular
(raio, textura, concavidade... em três versões: média, erro-padrão e "pior"). Não há valores ausentes
nem variáveis categóricas, então a limpeza se resume a verificar isso (feito em `tests/test_data.py`)."""),
code("""df = data.load_data()
print(df.shape); df.head()"""),

md("""## 2. Exploração (EDA)
**Balanceamento:** 63% benigno / 37% maligno — desbalanceamento moderado. Por isso não usamos só accuracy
(um modelo que chuta "benigno" acertaria 63%) e priorizamos **recall** da classe maligna."""),
code("""display(eda.class_balance(df))
eda.describe(df).round(3)"""),
code("""display(Image(eda.plot_distributions(df)))
display(Image(eda.plot_boxplots(df)))"""),
md("""**Discussão:** features de tamanho (`radius`, `perimeter`, `area`) e de forma (`concavity`, `concave points`)
separam bem as classes — malignos são maiores e mais irregulares. `smoothness`, `symmetry` e `fractal dimension`
quase não separam. Há outliers, mas são casos reais (tumores grandes), não erros — mantidos."""),
code("""display(eda.top_correlated_with_target(df).round(3))
display(Image(eda.plot_correlation(df)))"""),
md("""**Correlação:** `worst concave points` (0,79), `worst perimeter` (0,78) e `mean concave points` (0,78) lideram.
O heatmap mostra blocos com r≈0,99 (radius/perimeter/area): **multicolinearidade** forte. Consequências:
usamos regularização L2 na regressão logística e lemos importâncias individuais com cautela."""),

md("""## 3. Pré-processamento e separação
- Split **estratificado** 60/20/20 (treino 341 / validação 114 / teste 114), `random_state=42`.
- `StandardScaler` dentro de um `Pipeline`: ajustado só no treino → **sem data leakage**.
- Validação escolhe o modelo; teste é usado **uma única vez** no final."""),
code("""X_tr, X_va, X_te, y_tr, y_va, y_te = preprocessing.split_data(df)
pd.DataFrame({"treino": y_tr.value_counts(), "validacao": y_va.value_counts(), "teste": y_te.value_counts()})"""),

md("""## 4. Modelos e treinamento
Dois modelos simples e complementares: **Regressão Logística** (linear, interpretável, baseline) e
**Random Forest** (não linear, `class_weight="balanced"`). Métricas na **validação**:"""),
code("""train.train_all().round(4)"""),

md("""## 5. Avaliação no teste
**Por que estas métricas?** Recall = fração de malignos detectados (falso negativo é o erro mais grave:
atrasa tratamento). F1 equilibra recall e precisão (evita marcar tudo como maligno → biópsias desnecessárias).
Accuracy fica como visão geral."""),
code("""evaluate.evaluate_all().round(4)"""),
md("""**Resultado:** a Regressão Logística venceu na validação e confirmou no teste — recall 0,95 (2 falsos negativos
em 42 malignos), 1 falso positivo, F1 0,96. A Random Forest teve 4 falsos negativos. Modelo escolhido: **Regressão Logística**."""),

md("""## 6. Interpretação — Feature Importance e SHAP"""),
code("""display(explain.feature_importance().head(10).round(3))
display(Image(explain.plot_feature_importance()))"""),
code("""display(Image(explain.plot_shap_summary()))
display(Image(explain.plot_shap_individual(i=0)))"""),
md("""**Leitura:** valores altos de `worst texture`, `mean concave points`, `radius error` e `worst symmetry` empurram
para maligno; `compactness error` e `fractal dimension error` altos empurram para benigno. O ranking do modelo
difere da correlação simples (efeito da multicolinearidade: o peso se distribui entre features redundantes).
O gráfico *waterfall* explica **um paciente**: é o que o médico vê no app para julgar se a sugestão faz sentido."""),

md("""## 7. Discussão crítica — pode ser usado na prática?
**Como apoio, sim; como diagnóstico, não.**
- Mesmo o melhor modelo deixa passar ~5% dos malignos no teste; em triagem isso exige que o médico revise todo caso,
  especialmente os de probabilidade intermediária.
- O dataset é pequeno (569), de um único centro e antigo — pode não representar outra população/equipamento
  (*dataset shift*). Antes de uso real seria preciso validação externa e prospectiva.
- As features vêm de uma medição laboratorial padronizada; o modelo não substitui o exame, só organiza sua leitura.
- O SHAP dá transparência caso a caso, mas explica o **modelo**, não a biologia.
- Fluxo proposto: exame → modelo (probabilidade) → SHAP → **avaliação humana** → decisão final do médico."""),
]
nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("notebook gerado")
