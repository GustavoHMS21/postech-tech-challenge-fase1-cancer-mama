"""Interpretação dos modelos com feature importance e SHAP.

- Feature importance: relevância global de cada feature no modelo.
- SHAP: contribuição de cada feature para a previsão de cada paciente (explicação local),
  agregada no gráfico beeswarm para a visão global.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import load_data
from src.preprocessing import split_data
from src.train import MODELS_DIR

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
# Modelo selecionado pelo maior recall na validação; usado também pela interface.
BEST_MODEL = "logistic_regression"


def feature_importance() -> pd.DataFrame:
    """Importância de cada feature nos dois modelos, normalizada para soma 1.

    - Regressão Logística: valor absoluto dos coeficientes, comparáveis entre si
      porque as features estão padronizadas.
    - Random Forest: redução média de impureza (feature_importances_).
    """
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    lr_imp = np.abs(lr["model"].coef_[0])
    rf_imp = rf["model"].feature_importances_
    df = pd.DataFrame({"logistic_regression": lr_imp / lr_imp.sum(),
                       "random_forest": rf_imp / rf_imp.sum()}, index=lr.feature_names_in_)
    return df.sort_values("logistic_regression", ascending=False)


def plot_feature_importance(top: int = 15, out: Path = FIG_DIR / "05_feature_importance.png") -> Path:
    """Barras horizontais das features mais importantes, ordenadas pela Regressão Logística."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ax = feature_importance().head(top).iloc[::-1].plot.barh(figsize=(8, 6), width=0.8)
    ax.set_title(f"Top {top} features: importância normalizada")
    ax.set_xlabel("importância (soma = 1)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def build_explainer(model, background: pd.DataFrame):
    """Explainer SHAP com o treino padronizado completo como distribuição de referência.

    Por padrão o masker amostra 100 linhas aleatórias do background, o que torna o
    valor esperado e os valores SHAP dependentes do sorteio. Com max_samples igual ao
    tamanho do treino, todas as amostras são usadas e o resultado é determinístico.
    """
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.Explainer(model, masker)


def shap_values_test(model_name: str = BEST_MODEL):
    """Valores SHAP do modelo para o conjunto de teste.

    O explainer recebe o classificador e os dados já padronizados, que são a entrada
    real do modelo. shap.Explainer seleciona o algoritmo adequado (Linear para a
    regressão logística, Tree para a Random Forest).
    """
    pipe = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    X_tr, _, X_te, _, _, _ = split_data(load_data())
    cols = X_te.columns
    X_tr_s = pd.DataFrame(pipe["scaler"].transform(X_tr), columns=cols)
    X_te_s = pd.DataFrame(pipe["scaler"].transform(X_te), columns=cols, index=X_te.index)
    sv = build_explainer(pipe["model"], X_tr_s)(X_te_s)
    # Na Random Forest a saída tem uma dimensão por classe; mantém-se a classe maligna.
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    # Os valores exibidos nos gráficos passam a ser os originais (unidades do exame).
    # Os valores SHAP não são alterados.
    sv.data = X_te.values
    return sv


def plot_shap_summary(model_name: str = BEST_MODEL, out: Path = FIG_DIR / "06_shap_summary.png") -> Path:
    """Beeswarm: impacto de cada feature na previsão de todos os pacientes do teste."""
    out.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values_test(model_name), max_display=15, show=False)
    plt.title(f"SHAP: impacto de cada feature ({model_name})")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_shap_individual(i: int = 0, model_name: str = BEST_MODEL,
                         out: Path = FIG_DIR / "07_shap_paciente_exemplo.png") -> Path:
    """Waterfall de um paciente do teste: contribuição de cada feature a partir do valor esperado."""
    out.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.waterfall(shap_values_test(model_name)[i], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    print("Top 10 importâncias:\n", feature_importance().head(10).round(3), "\n")
    for p in (plot_feature_importance(), plot_shap_summary(), plot_shap_individual()):
        print(p.name)
