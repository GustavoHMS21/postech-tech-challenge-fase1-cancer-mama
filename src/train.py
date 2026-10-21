"""Treinamento dos modelos de classificação.

Dois modelos, escolhidos por serem simples e complementares:
- Regressão Logística: linear, rápida, coeficientes interpretáveis. Baseline clássico.
- Random Forest: não linear, captura interações, robusta a escala e outliers.
Ambos passam pelo mesmo pipeline (scaler + modelo) de `preprocessing.py`.

Fluxo: treina no TREINO -> compara na VALIDAÇÃO -> salva os dois em models/.
O conjunto de TESTE não é tocado aqui (só em evaluate.py).
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
    """Dicionário nome -> modelo (não treinado).

    - LogisticRegression: max_iter=1000 porque com 30 features o padrão (100) às vezes
      não converge; a regularização L2 padrão (C=1) ajuda com a multicolinearidade
      vista na EDA (radius/perimeter/area).
    - RandomForest: 200 árvores é suficiente para estabilizar em dataset pequeno;
      class_weight="balanced" penaliza mais errar a classe minoritária (maligno),
      alinhado à prioridade de recall.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    }


def compute_metrics(y_true, y_pred) -> dict:
    """As três métricas exigidas. Classe positiva = 1 (maligno)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_all() -> pd.DataFrame:
    """Treina cada modelo, avalia na validação e salva o pipeline treinado em models/.

    Retorna uma tabela com as métricas de validação — é ela que orienta a escolha
    do modelo final, sem olhar o teste.
    """
    X_tr, X_va, _, y_tr, y_va, _ = split_data(load_data())
    MODELS_DIR.mkdir(exist_ok=True)
    rows = []
    for name, model in get_models().items():
        pipe = build_pipeline(model).fit(X_tr, y_tr)
        # joblib é o formato padrão do sklearn para persistir pipelines (mais eficiente
        # que pickle para arrays numpy). O app e o evaluate carregam estes arquivos.
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
        rows.append({"modelo": name, **compute_metrics(y_va, pipe.predict(X_va))})
    results = pd.DataFrame(rows).set_index("modelo")
    results.to_csv(MODELS_DIR.parent / "reports" / "results" / "metricas_validacao.csv")
    return results


if __name__ == "__main__":
    print("Métricas na VALIDAÇÃO:\n", train_all().round(4))
