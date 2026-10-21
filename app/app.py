"""Interface de demonstração (Streamlit) — apoio ao diagnóstico.

Executar: `streamlit run app/app.py` (a partir da raiz do projeto).

O que faz: carrega o pipeline JÁ TREINADO (models/*.joblib — nunca retreina aqui),
recebe as medidas de um exame, mostra a sugestão do modelo com probabilidade e a
explicação SHAP daquele caso. O médico avalia e decide (human-in-the-loop).
Por que Streamlit? Interface web em Python puro, local e gratuita, sem API/banco —
suficiente para demonstrar o fluxo no vídeo.
"""
import sys
from pathlib import Path

# Garante que `src/` seja importável mesmo rodando `streamlit run app/app.py`.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.data import load_data
from src.explain import BEST_MODEL
from src.preprocessing import split_data
from src.train import MODELS_DIR

st.set_page_config(page_title="Apoio ao diagnóstico — câncer de mama", layout="wide")


@st.cache_resource  # carrega modelo e dados uma vez por sessão (evita reler a cada clique)
def load_assets():
    pipe = joblib.load(MODELS_DIR / f"{BEST_MODEL}.joblib")
    X_tr, _, X_te, _, _, y_te = split_data(load_data())
    # Explicador SHAP montado sobre o classificador com dados escalados (mesma lógica de explain.py).
    background = pd.DataFrame(pipe["scaler"].transform(X_tr), columns=X_tr.columns)
    explainer = shap.Explainer(pipe["model"], background)
    return pipe, explainer, X_te, y_te


pipe, explainer, X_te, y_te = load_assets()

st.title("Sistema de apoio ao diagnóstico — câncer de mama")
# Aviso fixo: exigência do enunciado (médico tem a palavra final).
st.warning("**Apoio à decisão, não diagnóstico.** A saída abaixo é uma sugestão estatística "
           "de um protótipo acadêmico. A avaliação e a decisão final são do profissional de saúde.")

# --- Entrada ---------------------------------------------------------------
# Usamos pacientes do conjunto de TESTE como ponto de partida: o modelo nunca os viu
# no treino, então a demonstração é honesta. O usuário pode ajustar os valores.
st.sidebar.header("Exame")
idx = st.sidebar.selectbox("Paciente do conjunto de teste", options=list(range(len(X_te))),
                           format_func=lambda i: f"#{i} (real: {'maligno' if y_te.iloc[i] else 'benigno'})")
base = X_te.iloc[idx]
st.sidebar.caption("Ajuste as 10 medidas 'mean' se quiser simular um caso; as demais mantêm o valor do paciente.")
valores = base.copy()
for col in [c for c in X_te.columns if c.startswith("mean")]:
    # Limites do slider vêm do próprio conjunto (min/max observados), para não permitir valores impossíveis.
    valores[col] = st.sidebar.slider(col, float(X_te[col].min()), float(X_te[col].max()),
                                     float(base[col]), format="%.3f")
X_in = pd.DataFrame([valores])

# --- Previsão ----------------------------------------------------------------
# predict_proba[:, 1] = probabilidade da classe 1 (maligno). Mostramos a probabilidade,
# não só o rótulo, para que o médico veja o grau de confiança.
prob = float(pipe.predict_proba(X_in)[0, 1])
pred = "MALIGNO" if prob >= 0.5 else "BENIGNO"

c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Sugestão do modelo")
    st.metric("Classificação sugerida", pred)
    st.metric("Probabilidade de maligno", f"{prob:.1%}")
    if 0.3 <= prob <= 0.7:
        # Faixa intermediária: o modelo está incerto — sinalizamos explicitamente.
        st.info("Probabilidade intermediária: caso de maior incerteza, revisão humana especialmente importante.")
    st.caption(f"Modelo: {BEST_MODEL} · recall no teste: 0,95 · 2 falsos negativos em 42 malignos")

with c2:
    st.subheader("Por que o modelo sugeriu isso? (SHAP)")
    X_s = pd.DataFrame(pipe["scaler"].transform(X_in), columns=X_in.columns)
    sv = explainer(X_s)
    sv.data = X_in.values  # exibe valores originais das medidas, não os escalados
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)
    st.caption("Barras vermelhas empurram para maligno; azuis para benigno. f(x) é o log-odds do modelo.")

# --- Human-in-the-loop -------------------------------------------------------
st.subheader("Avaliação do profissional de saúde")
decisao = st.radio("Decisão final", ["Concordo com a sugestão", "Discordo — solicitar exames adicionais",
                                    "Discordo — outro diagnóstico"], horizontal=True)
st.text_area("Observações clínicas")
st.caption("Este registro fica apenas na tela (protótipo): demonstra que a decisão é humana e documentada.")
