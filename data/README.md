# Dados

Dataset: **Breast Cancer Wisconsin (Diagnostic)** — público, 569 amostras, 30 atributos numéricos
calculados a partir de imagens de aspirado por agulha fina de massas mamárias.

- Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- UCI: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

O projeto carrega a cópia embutida no scikit-learn (`sklearn.datasets.load_breast_cancer`), então
não é necessário download. `python -m src.data` gera `data/breast_cancer.csv` para consulta.

Alvo: `target` = 1 maligno, 0 benigno (212 malignos / 357 benignos).
