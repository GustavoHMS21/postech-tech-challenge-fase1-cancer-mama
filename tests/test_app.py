"""Teste de execução da interface em modo headless (streamlit.testing)."""
from streamlit.testing.v1 import AppTest


def test_app_renderiza_sem_erro():
    at = AppTest.from_file("../app/app.py", default_timeout=60).run()
    assert not at.exception
    assert "Apoio à decisão" in at.warning[0].value
    assert any("Probabilidade de maligno" in m.label for m in at.metric)
