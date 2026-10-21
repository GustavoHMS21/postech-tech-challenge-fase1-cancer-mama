"""Testes de feature importance e SHAP."""
from src import explain


def test_importancias_somam_1_e_shap_tem_formato_do_teste():
    imp = explain.feature_importance()
    assert imp.shape == (30, 2)
    assert abs(imp.sum() - 1).max() < 1e-6
    sv = explain.shap_values_test()
    assert sv.values.shape == (114, 30)
