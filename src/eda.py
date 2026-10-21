"""Análise exploratória: estatísticas descritivas, balanceamento, distribuições e correlação.

Tabelas e figuras são gravadas em reports/ e reutilizadas pelo notebook e pelo relatório.
"""
from pathlib import Path

import matplotlib

# Backend sem interface gráfica, necessário para execução em Docker e pytest.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import TARGET, load_data

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RES_DIR = REPORTS_DIR / "results"

PALETTE = {0: "#4c72b0", 1: "#c44e52"}
LABELS = {0: "benigno", 1: "maligno"}

# As 30 features são 10 medidas em três estatísticas (mean, error, worst). Os gráficos
# de distribuição usam apenas as médias, já que as versões "worst" são altamente
# correlacionadas com elas e repetiriam a mesma informação.
MEAN_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension",
]


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas por feature (uma linha por feature)."""
    return df.drop(columns=[TARGET]).describe().T


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem e proporção de cada classe."""
    counts = df[TARGET].value_counts().rename({1: "maligno", 0: "benigno"})
    return pd.DataFrame({"n": counts, "proporcao": (counts / len(df)).round(3)})


def plot_class_balance(df: pd.DataFrame, out: Path = FIG_DIR / "01_balanceamento.png") -> Path:
    """Gráfico de barras com o número de amostras por classe."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    class_balance(df)["n"].plot.bar(ax=ax, color=[PALETTE[0], PALETTE[1]], rot=0)
    ax.set_title("Distribuição das classes")
    ax.set_ylabel("nº de amostras")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_distributions(df: pd.DataFrame, out: Path = FIG_DIR / "02_distribuicoes_mean.png") -> Path:
    """Histogramas das features 'mean' separados por classe.

    Cada classe é normalizada como densidade para que a forma das distribuições seja
    comparável apesar do desbalanceamento. As duas classes compartilham os mesmos
    intervalos (bins) para que as barras fiquem alinhadas.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), MEAN_FEATURES):
        bins = np.histogram_bin_edges(df[col], bins=25)
        for k in (0, 1):
            ax.hist(df.loc[df[TARGET] == k, col], bins=bins, density=True, alpha=0.5,
                    color=PALETTE[k], label=LABELS[k])
        ax.set_title(col, fontsize=10)
    axes[0, 0].legend()
    fig.suptitle("Distribuição das features 'mean' por classe")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_boxplots(df: pd.DataFrame, out: Path = FIG_DIR / "03_boxplots_mean.png") -> Path:
    """Boxplots das features 'mean' por classe (medianas e outliers)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), MEAN_FEATURES):
        bp = ax.boxplot([df.loc[df[TARGET] == k, col] for k in (0, 1)],
                        tick_labels=[LABELS[0], LABELS[1]], patch_artist=True, widths=0.6)
        for patch, k in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(PALETTE[k])
            patch.set_alpha(0.7)
        ax.set_title(col, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação de Pearson.

    Com o alvo binário, a correlação de Pearson equivale à ponto-bisserial.
    """
    return df.corr(numeric_only=True)


def top_correlated_with_target(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """As n features com maior correlação absoluta com o alvo."""
    corr = correlation(df)[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def plot_correlation(df: pd.DataFrame, out: Path = FIG_DIR / "04_correlacao.png") -> Path:
    """Heatmap da matriz de correlação entre todas as variáveis."""
    out.parent.mkdir(parents=True, exist_ok=True)
    corr = correlation(df)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title("Matriz de correlação (Pearson)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def run_eda() -> dict:
    """Gera todas as tabelas e figuras da EDA e retorna os caminhos."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    outputs = {
        "describe": RES_DIR / "eda_describe.csv",
        "balance": RES_DIR / "eda_balanceamento.csv",
        "top_corr": RES_DIR / "eda_top_correlacoes.csv",
    }
    describe(df).to_csv(outputs["describe"])
    class_balance(df).to_csv(outputs["balance"])
    top_correlated_with_target(df).rename("corr_com_target").to_csv(outputs["top_corr"])
    outputs["fig_balance"] = plot_class_balance(df)
    outputs["fig_dist"] = plot_distributions(df)
    outputs["fig_box"] = plot_boxplots(df)
    outputs["fig_corr"] = plot_correlation(df)
    return outputs


if __name__ == "__main__":
    df = load_data()
    print("Balanceamento:\n", class_balance(df), "\n")
    print("Top 10 correlações com o alvo:\n", top_correlated_with_target(df).round(3), "\n")
    for k, v in run_eda().items():
        print(f"{k:12s} {v.relative_to(REPORTS_DIR.parent)}")
