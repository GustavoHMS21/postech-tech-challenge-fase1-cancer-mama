"""Separação treino/validação/teste e pipeline de pré-processamento.

Ordem correta (evita data leakage):
  1) separar os conjuntos;  2) ajustar transformações SÓ no treino;  3) aplicar nos demais.
Aqui o passo 2 fica dentro de um `Pipeline` do sklearn, então o `.fit()` do modelo
ajusta o scaler e o classificador juntos, apenas com os dados de treino.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import TARGET

# Mesmo seed em todo o projeto: resultados reprodutíveis entre notebook, testes e app.
RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Divide em treino (60%), validação (20%) e teste (20%).

    - stratify=y mantém a proporção maligno/benigno igual nos três conjuntos; com
      classes desbalanceadas isso evita, por sorteio, um teste com poucos malignos.
    - Fazemos dois splits porque `train_test_split` só separa em dois: primeiro
      tiramos o teste (guardado até a avaliação final), depois tiramos a validação
      do que sobrou (usada para comparar modelos e escolher hiperparâmetros).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    # val_size é relativo ao total; ajustamos para a fração do que restou (0.2/0.8 = 0.25).
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(model) -> Pipeline:
    """Empacota scaler + modelo em um único objeto.

    Por que StandardScaler? As features têm escalas muito diferentes (area ~ 10³,
    fractal dimension ~ 10⁻²). Modelos baseados em distância/gradiente (regressão
    logística) precisam de escala comparável; para árvores é indiferente, mas manter
    o mesmo pipeline para os dois modelos simplifica o código e a comparação.
    Por que não há tratamento de nulos/categóricas? A EDA mostrou 0 valores ausentes
    e todas as 30 features são numéricas contínuas — adicionar imputer/encoder seria
    código morto.
    """
    return Pipeline([("scaler", StandardScaler()), ("model", model)])
