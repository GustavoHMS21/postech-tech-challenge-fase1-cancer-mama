"""Smoke test do app: garante que a página carrega e renderiza sem exceção.

`streamlit.testing.v1.AppTest` roda o script em modo headless, sem abrir navegador —
o jeito mais curto de validar o app dentro do pytest.
"""
from streamlit.testing.v1 import AppTest


def test_app_renderiza_sem_erro():
    at = AppTest.from_file("../app/app.py", default_timeout=60).run()
    assert not at.exception
    assert "Apoio à decisão" in at.warning[0].value  # aviso human-in-the-loop presente
    assert any("Probabilidade de maligno" in m.label for m in at.metric)
