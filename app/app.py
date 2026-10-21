"""Interface Streamlit para apoio ao diagnóstico.

Execução: streamlit run app/app.py (a partir da raiz do projeto).

Carrega o pipeline treinado em models/, recebe as medidas de um exame e exibe a
classificação sugerida, a probabilidade de malignidade e a explicação SHAP do caso.
A decisão final é registrada pelo profissional de saúde.
"""
import sys
from pathlib import Path

# Permite importar o pacote src quando o script é executado pelo Streamlit.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.data import load_data
from src.explain import BEST_MODEL, build_explainer
from src.preprocessing import split_data
from src.train import MODELS_DIR

st.set_page_config(page_title="Apoio ao diagnóstico: câncer de mama", layout="wide")


@st.cache_resource
def load_assets():
    """Carrega pipeline, explainer SHAP, conjunto de teste e métricas uma vez por sessão."""
    pipe = joblib.load(MODELS_DIR / f"{BEST_MODEL}.joblib")
    X_tr, _, X_te, _, _, y_te = split_data(load_data())
    background = pd.DataFrame(pipe["scaler"].transform(X_tr), columns=X_tr.columns)
    explainer = build_explainer(pipe["model"], background)
    metrics = pd.read_csv(ROOT / "reports" / "results" / "metricas_teste.csv", index_col=0).loc[BEST_MODEL]
    return pipe, explainer, X_te, y_te, metrics


pipe, explainer, X_te, y_te, metrics = load_assets()

st.title("Sistema de apoio ao diagnóstico: câncer de mama")
st.warning("**Apoio à decisão clínica.** O resultado é uma estimativa estatística e não constitui "
           "diagnóstico. A avaliação e a decisão final cabem ao profissional de saúde.")

# Entrada: pacientes do conjunto de teste, não utilizados no treinamento.
st.sidebar.header("Exame")
idx = st.sidebar.selectbox("Paciente do conjunto de teste", options=list(range(len(X_te))),
                           format_func=lambda i: f"#{i} (real: {'maligno' if y_te.iloc[i] else 'benigno'})")
base = X_te.iloc[idx]
st.sidebar.caption("As 10 medidas 'mean' podem ser ajustadas para simulação; as demais mantêm o valor do paciente.")
valores = base.copy()
for col in [c for c in X_te.columns if c.startswith("mean")]:
    # Limites restritos ao intervalo observado no conjunto de teste.
    valores[col] = st.sidebar.slider(col, float(X_te[col].min()), float(X_te[col].max()),
                                     float(base[col]), format="%.3f")
X_in = pd.DataFrame([valores])

# Previsão com limiar de 0,5 sobre a probabilidade da classe maligna.
prob = float(pipe.predict_proba(X_in)[0, 1])
pred = "MALIGNO" if prob >= 0.5 else "BENIGNO"

c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Sugestão do modelo")
    st.metric("Classificação sugerida", pred)
    # Valores abaixo de 0,1% são exibidos como limite, pois "0,0%" sugeriria certeza absoluta.
    st.metric("Probabilidade de maligno", "< 0.1%" if prob < 0.001 else f"{prob:.1%}")
    if 0.3 <= prob <= 0.7:
        st.info("Probabilidade intermediária (30% a 70%): maior incerteza do modelo, revisão prioritária.")
    st.caption(f"Modelo: {BEST_MODEL} | recall no teste: {metrics['recall']:.3f} | "
               f"falsos negativos: {int(metrics['fn'])} de {int(metrics['fn'] + metrics['tp'])} malignos")

with c2:
    st.subheader("Explicação da previsão (SHAP)")
    X_s = pd.DataFrame(pipe["scaler"].transform(X_in), columns=X_in.columns)
    sv = explainer(X_s)
    sv.data = X_in.values
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Barras vermelhas aumentam a probabilidade de maligno; azuis reduzem. "
               "f(x) está em log-odds.")

st.subheader("Avaliação do profissional de saúde")
decisao = st.radio("Decisão final", ["Concordo com a sugestão", "Discordo: solicitar exames adicionais",
                                    "Discordo: outro diagnóstico"], horizontal=True)
st.text_area("Observações clínicas")
st.caption("O registro não é persistido nesta versão; o campo representa a etapa de validação humana.")
