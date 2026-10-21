"""Configuração compartilhada dos testes (lida automaticamente pelo pytest).

Por que este arquivo existe?
O pytest roda os arquivos de teste em ordem alfabética. Assim, test_app.py e
test_explain.py rodam ANTES de test_train_evaluate.py, que é quem treina e salva os
modelos em models/. Num clone limpo do repositório (sem a pasta models/ preenchida),
esses dois testes falhariam por não encontrar os arquivos .joblib.

Este fixture treina os modelos uma única vez, no início da sessão de testes, e só
quando algum deles está faltando. Com os modelos já presentes, ele não faz nada
(mantém os testes rápidos).
"""
import pytest

from src.train import MODELS_DIR, get_models, train_all


@pytest.fixture(scope="session", autouse=True)
def garantir_modelos_treinados():
    """Garante que os pipelines .joblib existam antes de qualquer teste que os carrega."""
    # get_models() é a fonte única dos nomes; se um modelo novo for adicionado lá,
    # este teste passa a exigi-lo também, sem precisar editar esta lista à mão.
    esperados = [MODELS_DIR / f"{nome}.joblib" for nome in get_models()]
    if not all(arquivo.exists() for arquivo in esperados):
        train_all()
