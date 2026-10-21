# Tech Challenge — Fase 1 · Apoio ao diagnóstico de câncer de mama

Protótipo acadêmico (POSTECH) de um sistema de **apoio à decisão** que classifica tumores de mama como
**maligno** ou **benigno** com Machine Learning, a partir do dataset público *Breast Cancer Wisconsin (Diagnostic)*.

> O modelo produz uma **sugestão com probabilidade e explicação (SHAP)**. A decisão final é sempre do
> profissional de saúde (human-in-the-loop).

- Relatório técnico: [`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)
- Notebook de demonstração: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)
- Dataset: [`data/README.md`](data/README.md) (link Kaggle/UCI; cópia embutida no scikit-learn)

## Como executar

### Local (Python 3.10+)
```bash
pip install -r requirements.txt
python -m src.run_pipeline      # EDA + treino + avaliação + SHAP -> models/ e reports/
streamlit run app/app.py        # interface em http://localhost:8501
pytest -q                       # testes
```
Notebook: `cd notebooks && python build_notebook.py && python -m nbconvert --to notebook --execute --inplace analysis.ipynb`

### Docker
```bash
docker build -t tech-challenge .
docker run -p 8501:8501 tech-challenge   # abra http://localhost:8501
```
O build já treina os modelos e gera os relatórios (não há download: o dataset vem com o scikit-learn).

## Estrutura
```
DIRETRIZES.md         regras do projeto: arquitetura, código, data leakage, limites do modelo
src/data.py           carrega o dataset (1 = maligno, 0 = benigno)
src/eda.py            estatísticas, distribuições, correlação -> reports/
src/preprocessing.py  split estratificado 60/20/20 + Pipeline(StandardScaler, modelo)
src/train.py          Regressão Logística e Random Forest; métricas na validação; salva models/
src/evaluate.py       Accuracy, Recall, F1 e matriz de confusão no teste (usado uma única vez)
src/explain.py        Feature importance + SHAP (global e por paciente)
src/run_pipeline.py   executa tudo acima em sequência
app/app.py            Streamlit: previsão + probabilidade + SHAP + registro da decisão médica
notebooks/            analysis.ipynb (gerado por build_notebook.py; só chama src/)
tests/                pytest (13 testes)
reports/figures/      gráficos  ·  reports/results/  tabelas CSV
```

## Resultados (conjunto de teste, n = 114)

| Modelo | Accuracy | Recall (maligno) | F1 | Falsos negativos |
|---|---|---|---|---|
| **Regressão Logística** (escolhido) | 0,974 | 0,952 | 0,964 | 2 / 42 |
| Random Forest | 0,974 | 0,952 | 0,964 | 2 / 42 |

Métrica prioritária: **recall** da classe maligna (falso negativo = câncer não detectado). Detalhes,
gráficos e discussão crítica no relatório técnico.

> **Reprodutibilidade:** os números acima foram validados com Python 3.12 e as versões
> pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, shap 0.52.0, matplotlib 3.11.1,
> seaborn 0.13.2 e streamlit 1.63.0. Com versões diferentes das bibliotecas, a Random
> Forest pode apresentar pequenas variações; a escolha da Regressão Logística (maior
> recall na validação) se mantém.

## Fluxo do sistema
```
exame (30 medidas) -> pipeline (scaler + modelo) -> probabilidade de maligno
                   -> SHAP (por que) -> avaliação do médico -> decisão final
```
