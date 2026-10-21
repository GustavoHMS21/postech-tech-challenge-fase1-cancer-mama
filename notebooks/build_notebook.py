"""Gera notebooks/analysis.ipynb.

O notebook é mantido como script para facilitar a revisão no controle de versão e para
garantir que ele apenas chame as funções de src/, sem duplicar lógica.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Tech Challenge Fase 1: apoio ao diagnóstico de câncer de mama
Classificação de tumores de mama em maligno (1) ou benigno (0) com o dataset Breast Cancer Wisconsin (Diagnostic).
O resultado do modelo é uma sugestão de apoio à decisão; o diagnóstico cabe ao profissional de saúde.

O notebook segue a ordem do enunciado: dados, exploração, pré-processamento, modelagem, avaliação e interpretação.
A lógica está implementada e testada nos módulos de `src/`."""),
code("""import sys; sys.path.append("..")
import pandas as pd
from IPython.display import Image, display
from src import data, eda, preprocessing, train, evaluate, explain"""),

md("""## 1. Dados
569 exames de punção aspirativa por agulha fina, com 30 medidas numéricas do núcleo celular (10 características
em três estatísticas: média, erro-padrão e maior valor). Não há valores ausentes nem variáveis categóricas."""),
code("""df = data.load_data()
print(df.shape)
print("valores ausentes:", df.isna().sum().sum())
df.head()"""),

md("""## 2. Análise exploratória
Balanceamento de 63% benigno e 37% maligno. Um classificador que sempre prevê benigno teria accuracy de 62,7%,
por isso a avaliação prioriza recall e F1."""),
code("""display(eda.class_balance(df))
eda.describe(df).round(3)"""),
code("""display(Image(eda.plot_distributions(df)))
display(Image(eda.plot_boxplots(df)))"""),
md("""As features de tamanho (`radius`, `perimeter`, `area`) e de irregularidade do contorno (`concavity`,
`concave points`) separam bem as classes. `smoothness`, `symmetry` e `fractal dimension` têm sobreposição elevada.
Os outliers concentram-se na classe maligna e correspondem a tumores grandes, não a erros; foram mantidos."""),
code("""display(eda.top_correlated_with_target(df).round(3))
display(Image(eda.plot_correlation(df)))"""),
md("""Maiores correlações com o alvo: `worst concave points` (0,79), `worst perimeter` (0,78) e `mean concave points`
(0,78). Há multicolinearidade forte (r ≈ 0,99 entre radius, perimeter e area), o que motivou a regularização L2 na
regressão logística e a interpretação das importâncias por grupos de features."""),

md("""## 3. Pré-processamento e divisão dos dados
- Divisão estratificada 60/20/20 (341/114/114), `random_state=42`.
- `StandardScaler` dentro de um `Pipeline`, ajustado apenas no treino.
- Validação para seleção do modelo; teste usado uma única vez na avaliação final."""),
code("""X_tr, X_va, X_te, y_tr, y_va, y_te = preprocessing.split_data(df)
pd.DataFrame({"treino": y_tr.value_counts(), "validacao": y_va.value_counts(), "teste": y_te.value_counts()})"""),

md("""## 4. Modelos
Regressão Logística (linear, L2) e Random Forest (200 árvores, `class_weight="balanced"`). Métricas na validação:"""),
code("""train.train_all().round(4)"""),
md("""A Regressão Logística tem o maior recall na validação (42 de 43 malignos) e é selecionada."""),

md("""## 5. Avaliação no teste
Recall é a métrica prioritária: o falso negativo é o erro de maior custo clínico. F1 penaliza o excesso de falsos
positivos e accuracy é reportada como visão geral."""),
code("""display(evaluate.evaluate_all().round(4))
display(Image(evaluate.plot_confusion_matrices()))"""),
md("""Os dois modelos empatam no teste: 40 de 42 malignos detectados e 1 falso positivo. O intervalo de confiança de
95% para o recall (Wilson) vai de 0,84 a 0,99, então a diferença de um paciente na validação não indica
superioridade. A Regressão Logística é mantida por ser mais simples e interpretável."""),
code("""evaluate.error_analysis()"""),
md("""Os modelos erram pacientes diferentes. O paciente #16 é falso negativo nos dois e recebe probabilidade de 7% na
Regressão Logística, um caso maligno com perfil típico de benigno que não é capturado pela faixa de incerteza."""),

md("""## 6. Interpretação"""),
code("""display(explain.feature_importance().head(10).round(3))
display(Image(explain.plot_feature_importance()))"""),
code("""display(Image(explain.plot_shap_summary()))
display(Image(explain.plot_shap_individual(i=0)))"""),
md("""Valores altos de `worst texture`, `mean concave points`, `radius error`, `worst symmetry` e `worst radius`
aumentam a probabilidade de malignidade; `compactness error` e `fractal dimension error` altos a reduzem. O ranking
difere da correlação simples por efeito da multicolinearidade: o peso é distribuído entre features redundantes, e
os coeficientes negativos de algumas features de erro-padrão funcionam como termos de ajuste. O waterfall decompõe
a previsão de um paciente e é o formato exibido na interface."""),

md("""## 7. Discussão crítica
O modelo é adequado como apoio à triagem, sob supervisão médica, e não como diagnóstico automático.
- Cerca de 5% dos malignos não foram detectados no teste, um deles com probabilidade baixa.
- Dataset pequeno, de um único centro e da década de 1990; o uso real exige validação externa e prospectiva.
- O modelo considera apenas as medidas citológicas, sem histórico, exame físico ou imagem.
- O SHAP explica o comportamento do modelo, não a biologia do tumor.
- O limiar de 0,5 pode ser reduzido para aumentar o recall; essa escolha é clínica e institucional.

Fluxo: exame, modelo (probabilidade), SHAP, avaliação do profissional e decisão final."""),
]
nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("notebook gerado")
