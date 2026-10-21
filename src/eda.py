"""Análise exploratória (EDA) do dataset.

Objetivo: entender os dados ANTES de modelar — tamanho, tipos, balanceamento das
classes, distribuição das features e correlações. Tudo é salvo em `reports/`
para ser reaproveitado no relatório técnico e no notebook, sem recalcular.

Por que EDA em um módulo e não só no notebook? Para que as figuras sejam
reprodutíveis por um comando (`python -m src.eda`) e testáveis com pytest.
"""
from pathlib import Path

import matplotlib

# Backend "Agg" desenha em arquivo sem precisar de janela gráfica — necessário
# para rodar em servidor, Docker e pytest (ambientes sem display).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import TARGET, load_data

# Pastas de saída relativas ao projeto (mesma lógica de DATA_DIR em data.py).
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RES_DIR = REPORTS_DIR / "results"

# As 30 features são 10 medidas x 3 estatísticas (mean, error, worst). Para os
# histogramas usamos apenas as 10 "mean": são as mais interpretáveis clinicamente
# e evitam 30 gráficos redundantes (as versões "worst" são fortemente correlacionadas).
MEAN_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension",
]


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas (count, média, desvio, quartis) de cada feature.

    Transpomos (.T) para que cada linha seja uma feature — mais legível com 30 colunas.
    """
    return df.drop(columns=[TARGET]).describe().T


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem e proporção de cada classe.

    Importante para decidir se precisamos de balanceamento e para escolher métricas:
    com desbalanceamento (~63% benigno), accuracy sozinha engana — daí recall e F1.
    """
    counts = df[TARGET].value_counts().rename({1: "maligno", 0: "benigno"})
    return pd.DataFrame({"n": counts, "proporcao": (counts / len(df)).round(3)})


def plot_class_balance(df: pd.DataFrame, out: Path = FIG_DIR / "01_balanceamento.png") -> Path:
    """Gráfico de barras das classes — visualização direta do desbalanceamento."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    class_balance(df)["n"].plot.bar(ax=ax, color=["#4c72b0", "#c44e52"], rot=0)
    ax.set_title("Distribuição das classes")
    ax.set_ylabel("nº de amostras")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)  # libera memória; sem isso, muitas figuras acumulam em loops/pytest.
    return out


def plot_distributions(df: pd.DataFrame, out: Path = FIG_DIR / "02_distribuicoes_mean.png") -> Path:
    """Histogramas das 10 features 'mean', separados por classe.

    Por que separar por classe? Queremos ver quais features já distinguem maligno
    de benigno visualmente (ex.: 'mean concave points') — isso antecipa o que o
    modelo e o SHAP vão apontar como importante.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), MEAN_FEATURES):
        # kde=False e stat="density" para comparar formas mesmo com classes de tamanhos diferentes.
        sns.histplot(data=df, x=col, hue=TARGET, stat="density", common_norm=False,
                     bins=25, ax=ax, palette={0: "#4c72b0", 1: "#c44e52"}, alpha=0.5)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.get_legend().set_title("1=maligno")
    fig.suptitle("Distribuição das features 'mean' por classe")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_boxplots(df: pd.DataFrame, out: Path = FIG_DIR / "03_boxplots_mean.png") -> Path:
    """Boxplots das features 'mean' por classe — evidenciam outliers e separação de medianas."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), MEAN_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, palette={0: "#4c72b0", 1: "#c44e52"}, hue=TARGET, legend=False)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("0=benigno  1=maligno")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação de Pearson entre todas as features e o alvo.

    Pearson é adequado porque todas as variáveis são numéricas contínuas (e o alvo
    binário 0/1 funciona como correlação ponto-bisserial).
    """
    return df.corr(numeric_only=True)


def top_correlated_with_target(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """As n features mais correlacionadas (em módulo) com o alvo — ranking simples de relevância."""
    corr = correlation(df)[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def plot_correlation(df: pd.DataFrame, out: Path = FIG_DIR / "04_correlacao.png") -> Path:
    """Heatmap da correlação entre todas as variáveis.

    Serve para dois pontos do relatório: (1) quais features se relacionam com o alvo;
    (2) multicolinearidade — radius/perimeter/area são quase redundantes (r>0.99),
    o que justifica usar regularização na regressão logística e cuidado ao ler
    importâncias individuais.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation(df), cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title("Matriz de correlação (Pearson)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def run_eda() -> dict:
    """Executa toda a EDA e grava tabelas (CSV) e figuras (PNG). Retorna os caminhos gerados."""
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
        print(f"{k:12s} -> {v.relative_to(REPORTS_DIR.parent)}")
