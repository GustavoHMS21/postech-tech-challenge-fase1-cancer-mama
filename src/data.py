"""Carregamento do dataset Breast Cancer Wisconsin (Diagnostic).

Fonte original: UCI Machine Learning Repository, também publicado no Kaggle
(uciml/breast-cancer-wisconsin-data) — o link sugerido no enunciado.

Por que usamos a cópia embutida no scikit-learn (`load_breast_cancer`)?
- É exatamente o mesmo dataset (569 amostras, 30 atributos numéricos, mesmos valores).
- Não exige conta/credencial do Kaggle nem download manual, então o projeto roda
  em qualquer máquina e dentro do Docker sem passos extras (reprodutibilidade).
- Mesmo assim, salvamos um CSV em `data/` para cumprir a entrega "dataset ou link".
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Nome único da coluna alvo, importado pelos outros módulos para evitar strings repetidas.
TARGET = "target"
# Caminho absoluto de data/ calculado a partir deste arquivo, para funcionar
# independentemente da pasta de onde o script é executado (notebook, pytest, docker).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data() -> pd.DataFrame:
    """Retorna DataFrame com 30 features + coluna `target` (1 = maligno, 0 = benigno).

    Por que invertemos o alvo?
    No sklearn o alvo original é 0 = maligno, 1 = benigno. Em problemas médicos a
    convenção é "classe positiva = presença da doença". Com 1 = maligno, as métricas
    recall/precision/F1 do sklearn (que por padrão olham a classe 1) passam a medir
    diretamente o que importa: quantos casos malignos o modelo detecta.
    """
    # as_frame=True devolve um DataFrame pandas já com nomes de colunas legíveis.
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # 1 - target troca 0<->1 sem precisar de map/dicionário.
    df[TARGET] = 1 - df[TARGET]
    return df


def save_csv(df: pd.DataFrame, path: Path = DATA_DIR / "breast_cancer.csv") -> Path:
    """Salva uma cópia CSV do dataset (para o repositório/entrega)."""
    # mkdir garante que data/ exista mesmo em um clone limpo do repositório.
    path.parent.mkdir(parents=True, exist_ok=True)
    # index=False evita gravar a coluna de índice do pandas, que não é um atributo real.
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    # Execução direta (`python -m src.data`) serve como verificação rápida do carregamento.
    data = load_data()
    print(data.shape)
    print(data[TARGET].value_counts().rename({1: "maligno", 0: "benigno"}))
    print("salvo em", save_csv(data))
