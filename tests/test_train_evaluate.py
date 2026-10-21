"""Teste mínimo do treino e avaliação (roda o fluxo completo, ~segundos)."""
from src.evaluate import evaluate_all
from src.train import MODELS_DIR, train_all


def test_treino_salva_modelos_e_metricas_razoaveis():
    val = train_all()
    assert (MODELS_DIR / "logistic_regression.joblib").exists()
    assert (MODELS_DIR / "random_forest.joblib").exists()
    # Neste dataset qualquer modelo decente passa de 0.9; abaixo disso há bug (ex.: alvo trocado).
    assert (val["recall"] > 0.9).all()


def test_avaliacao_teste_consistente():
    test = evaluate_all()
    # tn+fp+fn+tp deve ser o tamanho do teste (114) para cada modelo
    assert (test[["tn", "fp", "fn", "tp"]].sum(axis=1) == 114).all()
    assert (test["f1"] > 0.9).all()
