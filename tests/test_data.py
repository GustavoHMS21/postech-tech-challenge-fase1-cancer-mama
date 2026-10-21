"""Testes de integridade do dataset e da codificação do alvo."""
from src.data import TARGET, load_data


def test_shape_e_colunas():
    df = load_data()
    assert df.shape == (569, 31)
    assert TARGET in df.columns


def test_sem_valores_ausentes():
    assert load_data().isna().sum().sum() == 0


def test_target_binario_com_maligno_como_1():
    df = load_data()
    assert set(df[TARGET].unique()) == {0, 1}
    # 212 malignos; sem a inversão do alvo a soma seria 357.
    assert df[TARGET].sum() == 212
