"""Treinamento dos modelos de classificação.

Os modelos são ajustados no conjunto de treino e comparados no conjunto de validação.
O conjunto de teste não é utilizado neste módulo.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data import load_data
from src.preprocessing import RANDOM_STATE, build_pipeline, split_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def get_models() -> dict:
    """Modelos avaliados, ainda não treinados.

    - Regressão Logística: modelo linear de referência, com coeficientes interpretáveis.
      A regularização L2 padrão (C=1) reduz o efeito da multicolinearidade observada
      na EDA; max_iter=1000 garante convergência do solver.
    - Random Forest: modelo não linear que captura interações entre features.
      class_weight="balanced" aumenta o peso dos erros na classe minoritária (maligno).
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, recall e F1, com a classe maligna (1) como positiva."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_all() -> pd.DataFrame:
    """Treina os modelos, salva os pipelines em models/ e retorna as métricas de validação."""
    X_tr, X_va, _, y_tr, y_va, _ = split_data(load_data())
    MODELS_DIR.mkdir(exist_ok=True)
    rows = []
    for name, model in get_models().items():
        pipe = build_pipeline(model).fit(X_tr, y_tr)
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
        rows.append({"modelo": name, **compute_metrics(y_va, pipe.predict(X_va))})
    results = pd.DataFrame(rows).set_index("modelo")
    results.to_csv(MODELS_DIR.parent / "reports" / "results" / "metricas_validacao.csv")
    return results


if __name__ == "__main__":
    print("Métricas na validação:\n", train_all().round(4))
