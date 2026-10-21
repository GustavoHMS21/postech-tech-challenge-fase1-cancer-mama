# Relatório Técnico — Tech Challenge Fase 1
## Sistema de apoio ao diagnóstico de câncer de mama com Machine Learning

**Repositório Git:** _[inserir link]_  **Vídeo de demonstração:** _[inserir link YouTube/Vimeo]_

---

## 1. Problema

Um hospital universitário quer acelerar a triagem e apoiar decisões clínicas. Nesta fase construímos a base
do sistema: um classificador que, a partir das medidas de um exame citológico (aspirado por agulha fina de
uma massa mamária), sugere se o tumor é **maligno** ou **benigno**, com probabilidade e explicação.

O sistema é **apoio à decisão**. Ele não diagnostica: organiza a leitura do exame, destaca os fatores mais
relevantes e entrega isso ao médico, que tem a palavra final.

## 2. Dataset

**Breast Cancer Wisconsin (Diagnostic)** — público (UCI / Kaggle `uciml/breast-cancer-wisconsin-data`).
569 exames, 30 atributos numéricos contínuos: 10 medidas do núcleo celular (raio, textura, perímetro,
área, suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal) em três versões
(média, erro-padrão, "pior"/maior valor). Alvo: 212 malignos (37,3%) e 357 benignos (62,7%).

Usamos a cópia embutida no scikit-learn (`load_breast_cancer`), idêntica à do Kaggle: elimina download e
credenciais, tornando o projeto reprodutível em qualquer máquina e no Docker. Convencionamos **1 = maligno**
(classe positiva = doença), para que recall e F1 meçam diretamente a detecção de câncer.

## 3. Exploração dos dados (EDA)

Figuras: `figures/01_balanceamento.png`, `02_distribuicoes_mean.png`, `03_boxplots_mean.png`, `04_correlacao.png`.
Tabelas: `results/eda_describe.csv`, `eda_balanceamento.csv`, `eda_top_correlacoes.csv`.

- **Qualidade:** nenhum valor ausente, nenhuma variável categórica, nenhum valor inconsistente (ex.: negativo).
  Verificado por teste automatizado.
- **Balanceamento:** desbalanceamento moderado (63/37). Consequência: accuracy sozinha engana — um modelo que
  sempre diz "benigno" acerta 63%. Por isso priorizamos recall e F1.
- **Distribuições:** as features de tamanho (`radius`, `perimeter`, `area`) e de forma (`concavity`,
  `concave points`) separam bem as classes: tumores malignos são maiores e mais irregulares. `smoothness`,
  `symmetry` e `fractal dimension` quase não separam. As distribuições são assimétricas à direita, com
  outliers que correspondem a tumores grandes reais — mantidos, pois não são erros de medição.
- **Correlação (Pearson):** maiores correlações com o alvo: `worst concave points` (0,79), `worst perimeter`
  (0,78), `mean concave points` (0,78), `worst radius` (0,78). O heatmap revela **multicolinearidade forte**
  (radius/perimeter/area com r ≈ 0,99; versões "mean" e "worst" da mesma medida muito correlacionadas).
  Isso orientou duas decisões: regularização L2 na regressão logística e cautela ao interpretar importâncias
  individuais.

## 4. Estratégia de pré-processamento

1. **Separação antes de tudo:** treino 60% (341) / validação 20% (114) / teste 20% (114), **estratificada**
   (mesma proporção de malignos nos três) e com `random_state=42` (reprodutível). A validação serve para
   comparar modelos; o teste é usado **uma única vez**, na avaliação final.
2. **Padronização:** `StandardScaler` (média 0, desvio 1), porque as escalas variam de 10⁻² (dimensão
   fractal) a 10³ (área) e a regressão logística é sensível a isso.
3. **Sem vazamento de dados:** o scaler vive dentro de um `Pipeline` do scikit-learn junto com o modelo, então
   é ajustado exclusivamente com o treino e aplicado aos demais conjuntos.
4. **Limpeza / imputação / encoding:** não necessários (EDA confirmou ausência de nulos e de categóricas);
   não adicionamos etapas que seriam código morto.

## 5. Modelos utilizados e por quê

| Modelo | Por quê |
|---|---|
| **Regressão Logística** (`max_iter=1000`, L2) | Baseline clássico para classificação binária; linear, rápido, coeficientes interpretáveis; a regularização L2 lida com a multicolinearidade. |
| **Random Forest** (200 árvores, `class_weight="balanced"`) | Não linear, captura interações, robusta a outliers; o peso balanceado penaliza mais o erro na classe minoritária (maligno). |

São técnicas simples e complementares (linear × ensemble de árvores), suficientes para o escopo do protótipo.

## 6. Métricas: escolha e justificativa

- **Recall (sensibilidade da classe maligna)** — métrica prioritária. Mede a fração de cânceres detectados.
  Um **falso negativo** (maligno classificado como benigno) atrasa o tratamento: é o erro clinicamente mais grave.
- **F1-score** — média harmônica de recall e precisão. Impede a solução trivial "marcar tudo como maligno"
  (recall 100%, mas inundaria o hospital de biópsias desnecessárias).
- **Accuracy** — visão geral, reportada por exigência, mas lida com cautela dado o desbalanceamento.

## 7. Resultados

**Validação (n = 114)** — usada para escolher o modelo:

| Modelo | Accuracy | Recall | F1 |
|---|---|---|---|
| Regressão Logística | 0,974 | **0,977** | 0,966 |
| Random Forest | 0,974 | 0,953 | 0,965 |

**Teste (n = 114, 42 malignos)** — avaliação final, uma única vez:

| Modelo | Accuracy | Recall | F1 | VN | FP | FN | VP |
|---|---|---|---|---|---|---|---|
| **Regressão Logística** | **0,974** | **0,952** | **0,964** | 71 | 1 | **2** | 40 |
| Random Forest | 0,956 | 0,905 | 0,938 | 71 | 1 | 4 | 38 |

A Regressão Logística venceu na validação (maior recall) e confirmou no teste: detectou 40 dos 42 malignos
(2 falsos negativos) com apenas 1 falso positivo. Foi o **modelo escolhido**. Um modelo mais simples superar
a Random Forest é coerente com o dataset: pequeno, com classes quase linearmente separáveis após padronização.

## 8. Interpretação dos resultados

Figuras: `figures/05_feature_importance.png`, `06_shap_summary.png`, `07_shap_paciente_exemplo.png`.

**Feature importance (global).** Para a regressão logística usamos o módulo dos coeficientes (comparáveis
porque as features estão padronizadas); para a floresta, a redução média de impureza. Ambas normalizadas.
Destaques na regressão logística: `worst texture`, `mean concave points`, `radius error`, `worst symmetry`,
`worst radius`, `worst area`. Na floresta: `mean concave points`, `worst area`, `worst radius`.

**SHAP (global e local).** O gráfico *beeswarm* mostra, para cada paciente do teste, quanto cada feature
empurrou a previsão. Valores altos de `worst texture`, `mean concave points`, `radius error` e `worst symmetry`
empurram para **maligno**; valores altos de `compactness error` e `fractal dimension error` empurram para
**benigno**. O gráfico *waterfall* explica **um paciente**: parte da previsão média e soma a contribuição de
cada medida até a previsão final — é exatamente o que a interface mostra ao médico.

**Observação importante.** O ranking do modelo difere da correlação simples da EDA (`worst concave points`
liderava a correlação, mas tem peso menor no modelo). É efeito da multicolinearidade: quando várias features
carregam a mesma informação, o modelo distribui o peso entre elas. Por isso a interpretação deve ser feita
por *grupos* de features (tamanho, irregularidade do contorno, textura), não por variáveis isoladas.

## 9. Discussão crítica — o modelo pode ser usado na prática?

**Como apoio à triagem, com supervisão médica: sim. Como diagnóstico automático: não.**

- **Erros residuais.** Mesmo o melhor modelo deixou passar 2 de 42 malignos (~5%) no teste. Em triagem, isso
  exige que todo caso seja revisado pelo médico, com atenção redobrada aos de probabilidade intermediária —
  a interface sinaliza explicitamente a faixa 30–70%.
- **Representatividade.** 569 exames, de um único centro, coletados há décadas. Outra população, outro
  equipamento ou outro protocolo de medição podem degradar o desempenho (*dataset shift*). Uso real exigiria
  validação externa e prospectiva, além de aprovação regulatória.
- **O que o modelo não vê.** Ele usa só as 30 medidas citológicas; não considera histórico, exame clínico,
  imagem ou outros exames. É uma peça do quadro, não o quadro.
- **Explicabilidade.** O SHAP dá transparência caso a caso e permite ao médico checar se a sugestão é coerente
  com o que ele observa — mas explica o **modelo**, não a biologia do tumor.
- **Limiar de decisão.** Usamos 0,5. Em um cenário real, o hospital poderia baixar o limiar para aumentar o
  recall (menos falsos negativos) ao custo de mais falsos positivos; essa é uma decisão clínica e institucional,
  não do modelo.

**Fluxo proposto (human-in-the-loop):**

```
exame -> modelo (classe + probabilidade) -> SHAP (fatores) -> avaliação do profissional -> decisão final
```

A interface Streamlit materializa esse fluxo: exibe a sugestão, a probabilidade, a explicação e um campo de
decisão/observações do profissional, com o aviso permanente de que se trata de apoio, não de diagnóstico.

## 10. Reprodutibilidade e organização

- Código modular em `src/` com comentários explicando cada decisão; 13 testes automatizados (`pytest`).
- `python -m src.run_pipeline` reproduz todos os resultados; `Dockerfile` empacota treino + interface.
- `DIRETRIZES.md` documenta as regras do projeto: padrões de código, cuidados com data leakage, limites do
  modelo e processo incremental (entender → planejar → implementar → validar → documentar).

## Anexos (prints e gráficos)

_Inserir aqui as figuras de `reports/figures/` e prints da interface (`streamlit run app/app.py`)._
