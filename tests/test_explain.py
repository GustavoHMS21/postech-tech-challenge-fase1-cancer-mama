"""Teste mínimo da explicabilidade (assume modelos já treinados por test_train_evaluate)."""
from src import explain


def test_importancias_somam_1_e_shap_tem_formato_do_teste():
    imp = explain.feature_importance()
    assert imp.shape == (30, 2)
    assert abs(imp.sum() - 1).max() < 1e-6  # normalização
    sv = explain.shap_values_test()
    assert sv.values.shape == (114, 30)  # um valor SHAP por paciente do teste x feature
