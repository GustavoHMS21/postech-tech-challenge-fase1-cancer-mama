"""Carregamento do dataset Breast Cancer Wisconsin (Diagnostic).

Fonte: UCI Machine Learning Repository (também disponível no Kaggle em
uciml/breast-cancer-wisconsin-data). Utiliza-se a cópia distribuída com o
scikit-learn, idêntica à original (569 amostras, 30 atributos), o que dispensa
download e credenciais e mantém a execução reprodutível, inclusive no Docker.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

TARGET = "target"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data() -> pd.DataFrame:
    """Retorna as 30 features e a coluna alvo (1 = maligno, 0 = benigno).

    No scikit-learn a codificação original é 0 = maligno. O alvo é invertido para
    que a classe positiva corresponda à doença; assim recall, precisão e F1, que
    por padrão avaliam a classe 1, medem diretamente a detecção de malignidade.
    """
    df = load_breast_cancer(as_frame=True).frame.copy()
    df[TARGET] = 1 - df[TARGET]
    return df


def save_csv(df: pd.DataFrame, path: Path = DATA_DIR / "breast_cancer.csv") -> Path:
    """Salva uma cópia do dataset em CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    data = load_data()
    print(data.shape)
    print(data[TARGET].value_counts().rename({1: "maligno", 0: "benigno"}))
    print("salvo em", save_csv(data))
