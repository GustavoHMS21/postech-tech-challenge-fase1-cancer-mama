"""Avaliação final no conjunto de teste.

Executada uma única vez, após a escolha do modelo pela validação, para que o teste
permaneça uma estimativa não enviesada do desempenho em dados novos.

Métricas (classe positiva = maligno):
- Recall: fração dos casos malignos detectados. Métrica prioritária, pois o falso
  negativo (maligno classificado como benigno) é o erro de maior custo clínico.
- F1: média harmônica entre precisão e recall; penaliza a estratégia de aumentar o
  recall classificando casos em excesso como malignos.
- Accuracy: visão geral, interpretada com cautela devido ao desbalanceamento
  (um classificador que sempre prevê benigno atinge cerca de 63%).
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data import load_data
from src.preprocessing import split_data
from src.train import MODELS_DIR, compute_metrics, get_models

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RES_DIR = REPORTS_DIR / "results"
FIG_DIR = REPORTS_DIR / "figures"


def evaluate_all() -> pd.DataFrame:
    """Calcula métricas e matriz de confusão de cada modelo salvo no conjunto de teste."""
    _, _, X_te, _, _, y_te = split_data(load_data())
    rows = []
    for name in get_models():
        pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
        y_pred = pipe.predict(X_te)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        rows.append({"modelo": name, **compute_metrics(y_te, y_pred),
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    results = pd.DataFrame(rows).set_index("modelo")
    RES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RES_DIR / "metricas_teste.csv")
    return results


def error_analysis() -> pd.DataFrame:
    """Lista os falsos negativos e falsos positivos de cada modelo no teste, com a probabilidade prevista.

    O índice "paciente" é a posição no conjunto de teste, a mesma usada na interface.
    """
    _, _, X_te, _, _, y_te = split_data(load_data())
    rows = []
    for name in get_models():
        prob = joblib.load(MODELS_DIR / f"{name}.joblib").predict_proba(X_te)[:, 1]
        for i, (real, p) in enumerate(zip(y_te, prob)):
            pred = int(p >= 0.5)
            if pred != real:
                rows.append({"modelo": name, "paciente": i, "real": int(real),
                             "tipo": "FN" if real == 1 else "FP", "prob_maligno": round(float(p), 3)})
    results = pd.DataFrame(rows)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RES_DIR / "erros_teste.csv", index=False)
    return results


def plot_confusion_matrices(out: Path = FIG_DIR / "08_matriz_confusao.png") -> Path:
    """Matrizes de confusão dos dois modelos no conjunto de teste, lado a lado."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _, _, X_te, _, _, y_te = split_data(load_data())
    names = list(get_models())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    for ax, name in zip(axes, names):
        pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
        ConfusionMatrixDisplay.from_estimator(
            pipe, X_te, y_te, display_labels=["benigno", "maligno"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Métricas no teste (n=114):\n", evaluate_all().round(4))
    print("Erros no teste:\n", error_analysis())
    print(plot_confusion_matrices().name)
