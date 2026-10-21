"""Testes do carregamento de dados.

Garantem que o dataset chegou íntegro e com o alvo na convenção do projeto
(1 = maligno). Se alguém trocar a fonte dos dados, estes testes acusam.
"""
from src.data import TARGET, load_data


def test_shape_e_colunas():
    # 569 amostras e 30 features + 1 alvo é a dimensão conhecida do dataset.
    df = load_data()
    assert df.shape == (569, 31)
    assert TARGET in df.columns


def test_sem_valores_ausentes():
    # O dataset original não tem NaN; se aparecer, o pré-processamento precisaria mudar.
    assert load_data().isna().sum().sum() == 0


def test_target_binario_com_maligno_como_1():
    df = load_data()
    assert set(df[TARGET].unique()) == {0, 1}
    # A soma do alvo é a contagem de malignos. O dataset tem 212 malignos e 357 benignos;
    # se a inversão do alvo fosse esquecida, a soma daria 357 e o teste falharia.
    assert df[TARGET].sum() == 212
