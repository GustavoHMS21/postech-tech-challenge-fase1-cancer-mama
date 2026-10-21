"""Interpretação dos modelos: Feature Importance e SHAP.

Duas visões complementares, ambas exigidas pelo enunciado:
- Feature importance (global): quais features pesam mais no modelo como um todo.
- SHAP (global e local): quanto cada feature empurrou a previsão de UM paciente para
  maligno ou benigno. É o que o médico vê no app para julgar se a sugestão faz sentido.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # mesmo motivo do eda.py: gerar PNG sem display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import load_data
from src.preprocessing import split_data
from src.train import MODELS_DIR

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
# Modelo vencedor na validação (maior recall). Centralizado aqui para app/notebook reutilizarem.
BEST_MODEL = "logistic_regression"


def feature_importance() -> pd.DataFrame:
    """Importância global de cada feature nos dois modelos, numa tabela só.

    - Regressão Logística: usamos |coeficiente|. Como o scaler padroniza todas as
      features, os coeficientes ficam na mesma escala e são comparáveis entre si.
    - Random Forest: `feature_importances_` (redução média de impureza).
    Normalizamos ambas para somar 1, senão as escalas seriam incomparáveis no gráfico.
    """
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    features = lr.feature_names_in_
    lr_imp = np.abs(lr["model"].coef_[0])
    rf_imp = rf["model"].feature_importances_
    df = pd.DataFrame({"logistic_regression": lr_imp / lr_imp.sum(),
                       "random_forest": rf_imp / rf_imp.sum()}, index=features)
    return df.sort_values("logistic_regression", ascending=False)


def plot_feature_importance(top: int = 15, out: Path = FIG_DIR / "05_feature_importance.png") -> Path:
    """Barras horizontais das `top` features mais importantes (ordenadas pela LR)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ax = feature_importance().head(top).iloc[::-1].plot.barh(figsize=(8, 6), width=0.8)
    ax.set_title(f"Top {top} features — importância normalizada")
    ax.set_xlabel("importância (soma = 1)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def shap_values_test(model_name: str = BEST_MODEL):
    """Calcula os valores SHAP do modelo no conjunto de TESTE.

    O SHAP precisa receber os dados já escalados (é o que o classificador enxerga),
    por isso aplicamos `pipe["scaler"].transform` antes e explicamos só `pipe["model"]`.
    `shap.Explainer` escolhe automaticamente o algoritmo: Linear para a regressão
    logística (exato, instantâneo) e Tree para a random forest. Os dados de treino
    escalados servem de "background" (referência do que é um paciente típico).
    """
    pipe = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    X_tr, _, X_te, _, _, _ = split_data(load_data())
    cols = X_te.columns
    X_tr_s = pd.DataFrame(pipe["scaler"].transform(X_tr), columns=cols)
    X_te_s = pd.DataFrame(pipe["scaler"].transform(X_te), columns=cols, index=X_te.index)
    explainer = shap.Explainer(pipe["model"], X_tr_s)
    sv = explainer(X_te_s)
    # Para a RF o shap devolve uma dimensão extra por classe; ficamos com a classe 1 (maligno).
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    # Trocamos os dados escalados pelos originais só para exibição: no gráfico o médico
    # quer ver "raio = 17.5", não "raio = 1.2 desvios-padrão". Os valores SHAP não mudam.
    sv.data = X_te.values
    return sv


def plot_shap_summary(model_name: str = BEST_MODEL, out: Path = FIG_DIR / "06_shap_summary.png") -> Path:
    """Beeswarm global: cada ponto é um paciente; cor = valor da feature, x = impacto na previsão."""
    out.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values_test(model_name), max_display=15, show=False)
    plt.title(f"SHAP — impacto de cada feature ({model_name})")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_shap_individual(i: int = 0, model_name: str = BEST_MODEL,
                         out: Path = FIG_DIR / "07_shap_paciente_exemplo.png") -> Path:
    """Waterfall de UM paciente do teste: mostra como cada feature levou da média à previsão.

    É o formato reutilizado no app: explicação local, caso a caso, para o médico avaliar.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(shap_values_test(model_name)[i], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    print("Top 10 importâncias:\n", feature_importance().head(10).round(3), "\n")
    for p in (plot_feature_importance(), plot_shap_summary(), plot_shap_individual()):
        print("->", p.name)
