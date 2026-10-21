"""Fixtures compartilhadas pelos testes."""
import pytest

from src.train import MODELS_DIR, get_models, train_all


@pytest.fixture(scope="session", autouse=True)
def garantir_modelos_treinados():
    """Treina os modelos caso algum .joblib esteja ausente.

    O pytest executa os arquivos em ordem alfabética, então test_app.py e
    test_explain.py rodam antes de test_train_evaluate.py. Em um clone limpo
    (models/ vazio), esses testes falhariam sem esta fixture.
    """
    esperados = [MODELS_DIR / f"{nome}.joblib" for nome in get_models()]
    if not all(arquivo.exists() for arquivo in esperados):
        train_all()
