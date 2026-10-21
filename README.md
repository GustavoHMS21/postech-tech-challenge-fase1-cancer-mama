# Tech Challenge Fase 1: apoio ao diagnóstico de câncer de mama

Projeto da Fase 1 do Tech Challenge (POSTECH). Classificação de tumores de mama em maligno ou benigno a partir do
dataset público Breast Cancer Wisconsin (Diagnostic), com interpretação por feature importance e SHAP e interface
de apoio à decisão clínica.

O sistema retorna uma sugestão com probabilidade e explicação. A decisão final cabe ao profissional de saúde.

- Relatório técnico: [`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)
- Notebook: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)
- Dataset: [`data/README.md`](data/README.md)

## Execução

### Local (Python 3.11 ou superior)

```bash
pip install -r requirements.txt
python -m src.run_pipeline      # EDA, treino, avaliação e SHAP; gera models/ e reports/
streamlit run app/app.py        # interface em http://localhost:8501
pytest -q                       # testes
```

Notebook:

```bash
cd notebooks
python build_notebook.py
python -m nbconvert --to notebook --execute --inplace analysis.ipynb
```

### Docker

```bash
docker build -t tech-challenge .
docker run -p 8501:8501 tech-challenge
```

A interface fica disponível em http://localhost:8501. O build executa o pipeline completo; não há download de
dados, pois o dataset é distribuído com o scikit-learn.

## Estrutura

```
src/data.py           carregamento do dataset (1 = maligno, 0 = benigno)
src/eda.py            estatísticas descritivas, distribuições e correlação
src/preprocessing.py  divisão estratificada 60/20/20 e Pipeline (StandardScaler + modelo)
src/train.py          Regressão Logística e Random Forest; métricas na validação
src/evaluate.py       métricas, matriz de confusão e análise de erros no teste
src/explain.py        feature importance e SHAP (global e por paciente)
src/run_pipeline.py   execução de todas as etapas em sequência
app/app.py            interface Streamlit: previsão, probabilidade, SHAP e registro da decisão
notebooks/            analysis.ipynb, gerado por build_notebook.py a partir das funções de src/
tests/                testes automatizados (pytest)
reports/figures/      gráficos
reports/results/      tabelas CSV
```

## Resultados no conjunto de teste (n = 114, 42 malignos)

| Modelo | Accuracy | Recall (maligno) | F1 | Falsos negativos |
|---|---|---|---|---|
| Regressão Logística (selecionado) | 0,974 | 0,952 | 0,964 | 2 |
| Random Forest | 0,974 | 0,952 | 0,964 | 2 |

O modelo foi selecionado pelo recall na validação (0,977 contra 0,953). No teste os dois modelos empatam; a
Regressão Logística é mantida por ser mais simples e interpretável. A análise completa está no relatório técnico.

Resultados obtidos com Python 3.12 e as versões fixadas em `requirements.txt`.

## Fluxo

```
exame (30 medidas) -> pipeline (scaler + modelo) -> probabilidade de maligno
                   -> SHAP -> avaliação do profissional -> decisão final
```
