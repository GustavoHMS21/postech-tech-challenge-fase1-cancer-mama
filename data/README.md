# Dados

Breast Cancer Wisconsin (Diagnostic): 569 amostras e 30 atributos numéricos calculados a partir de imagens
digitalizadas de punção aspirativa por agulha fina (PAAF) de massas mamárias.

- Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- UCI: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

O projeto utiliza a cópia distribuída com o scikit-learn (`sklearn.datasets.load_breast_cancer`), sem necessidade
de download. O arquivo `breast_cancer.csv` é gerado por `python -m src.data`.

Alvo: `target` = 1 para maligno (212 amostras) e 0 para benigno (357 amostras).
