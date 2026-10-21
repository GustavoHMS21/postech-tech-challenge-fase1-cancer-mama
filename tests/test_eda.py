"""Testes da EDA: verificam formatos das tabelas e que as figuras são geradas.

Usamos `tmp_path` do pytest para não sujar reports/ durante os testes.
"""
from src.data import TARGET, load_data
from src import eda


def test_describe_tem_30_features():
    df = load_data()
    assert eda.describe(df).shape[0] == 30  # uma linha por feature, alvo excluído


def test_class_balance_soma_total():
    df = load_data()
    bal = eda.class_balance(df)
    assert bal["n"].sum() == len(df)
    assert abs(bal["proporcao"].sum() - 1) < 0.01  # proporções somam ~1 (arredondamento)


def test_top_correlacoes_conhecidas():
    # 'worst concave points' e 'worst perimeter' são sabidamente as mais correlacionadas
    # com malignidade neste dataset; se sumirem do top 3, algo mudou nos dados.
    top = eda.top_correlated_with_target(load_data(), n=3)
    assert "worst concave points" in top.index
    assert (top > 0).all()  # sinal positivo: valores maiores -> mais maligno (alvo=1)


def test_figuras_sao_geradas(tmp_path):
    df = load_data()
    for fn in (eda.plot_class_balance, eda.plot_distributions, eda.plot_boxplots, eda.plot_correlation):
        out = fn(df, out=tmp_path / f"{fn.__name__}.png")
        assert out.exists() and out.stat().st_size > 0
