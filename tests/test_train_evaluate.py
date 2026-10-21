"""Testes do treinamento e da avaliação."""
from src.evaluate import error_analysis, evaluate_all
from src.train import MODELS_DIR, train_all


def test_treino_salva_modelos_e_metricas_razoaveis():
    val = train_all()
    assert (MODELS_DIR / "logistic_regression.joblib").exists()
    assert (MODELS_DIR / "random_forest.joblib").exists()
    # Recall abaixo de 0,9 neste dataset indica erro (por exemplo, alvo invertido).
    assert (val["recall"] > 0.9).all()


def test_avaliacao_teste_consistente():
    test = evaluate_all()
    assert (test[["tn", "fp", "fn", "tp"]].sum(axis=1) == 114).all()
    assert (test["f1"] > 0.9).all()


def test_analise_de_erros_bate_com_matriz_de_confusao():
    test = evaluate_all()
    erros = error_analysis()
    for name, row in test.iterrows():
        tipos = erros[erros["modelo"] == name]["tipo"].value_counts()
        assert tipos.get("FN", 0) == row["fn"] and tipos.get("FP", 0) == row["fp"]
