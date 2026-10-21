"""Testes das tabelas e figuras da EDA (figuras gravadas em diretório temporário)."""
from src.data import load_data
from src import eda


def test_describe_tem_30_features():
    assert eda.describe(load_data()).shape[0] == 30


def test_class_balance_soma_total():
    df = load_data()
    bal = eda.class_balance(df)
    assert bal["n"].sum() == len(df)
    assert abs(bal["proporcao"].sum() - 1) < 0.01


def test_top_correlacoes_conhecidas():
    top = eda.top_correlated_with_target(load_data(), n=3)
    assert "worst concave points" in top.index
    assert (top > 0).all()


def test_figuras_sao_geradas(tmp_path):
    df = load_data()
    for fn in (eda.plot_class_balance, eda.plot_distributions, eda.plot_boxplots, eda.plot_correlation):
        out = fn(df, out=tmp_path / f"{fn.__name__}.png")
        assert out.exists() and out.stat().st_size > 0
