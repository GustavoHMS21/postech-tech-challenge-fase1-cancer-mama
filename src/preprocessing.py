"""Separação treino/validação/teste e pipeline de pré-processamento.

A divisão é feita antes de qualquer ajuste. O StandardScaler fica dentro de um
Pipeline junto com o classificador, de modo que seus parâmetros (média e desvio)
são estimados apenas com o conjunto de treino, evitando vazamento de dados.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import TARGET

RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Divide os dados em treino (60%), validação (20%) e teste (20%), estratificado pelo alvo.

    São feitas duas chamadas a train_test_split: a primeira separa o teste e a segunda
    separa a validação do restante. A estratificação mantém a proporção de malignos
    igual nos três conjuntos.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    # val_size é relativo ao total; converte-se para a fração do conjunto restante (0,2 / 0,8 = 0,25).
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(model) -> Pipeline:
    """Pipeline com padronização (StandardScaler) seguida do classificador.

    As features variam de 10^-2 (fractal dimension) a 10^3 (area), e a regressão
    logística com regularização L2 é sensível à escala. Para a Random Forest a
    padronização é indiferente, mas o mesmo pipeline é mantido para os dois modelos.
    Não há imputação nem codificação: o dataset não possui valores ausentes nem
    variáveis categóricas.
    """
    return Pipeline([("scaler", StandardScaler()), ("model", model)])
