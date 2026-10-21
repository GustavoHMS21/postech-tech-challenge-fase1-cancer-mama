"""Testes da divisão dos dados e do pipeline."""
from sklearn.linear_model import LogisticRegression

from src.data import load_data
from src.preprocessing import build_pipeline, split_data


def test_split_tamanhos_e_estratificacao():
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(load_data())
    assert (len(X_tr), len(X_va), len(X_te)) == (341, 114, 114)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02 and abs(y_va.mean() - y_te.mean()) < 0.02


def test_pipeline_treina_e_prediz():
    X_tr, X_va, _, y_tr, _, _ = split_data(load_data())
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    # Scaler ajustado somente no treino: média ~0 no próprio treino.
    assert abs(pipe["scaler"].transform(X_tr).mean()) < 1e-6
    assert set(pipe.predict(X_va)) <= {0, 1}
