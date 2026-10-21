"""Avaliação final no conjunto de TESTE.

Executar apenas uma vez, depois de escolher o modelo pela validação. Se o teste for
usado para escolher/ajustar, ele deixa de ser uma estimativa honesta do desempenho
em dados novos.

Por que Accuracy, Recall e F1?
- Accuracy: visão geral, mas enganosa com desbalanceamento (62% benigno -> um modelo
  que chuta "benigno" sempre teria 62%).
- Recall (sensibilidade da classe maligna): fração de cânceres detectados. É a métrica
  prioritária — um falso negativo (maligno dito benigno) atrasa tratamento; é o erro
  mais grave clinicamente.
- F1: equilíbrio entre recall e precisão; evita "maximizar recall" marcando tudo como
  maligno, o que geraria biópsias desnecessárias.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data import load_data
from src.preprocessing import split_data
from src.train import MODELS_DIR, compute_metrics, get_models

RES_DIR = Path(__file__).resolve().parents[1] / "reports" / "results"


def evaluate_all() -> pd.DataFrame:
    """Carrega os pipelines salvos e calcula métricas + matriz de confusão no teste."""
    _, _, X_te, _, _, y_te = split_data(load_data())  # mesmo seed -> mesmo teste do treino
    rows = []
    for name in get_models():
        pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
        y_pred = pipe.predict(X_te)
        # ravel() transforma a matriz 2x2 em [tn, fp, fn, tp]; fn é o número que mais importa.
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        rows.append({"modelo": name, **compute_metrics(y_te, y_pred),
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    results = pd.DataFrame(rows).set_index("modelo")
    RES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RES_DIR / "metricas_teste.csv")
    return results


if __name__ == "__main__":
    print("Métricas no TESTE (n=114):\n", evaluate_all().round(4))
