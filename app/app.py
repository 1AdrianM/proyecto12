"""
Interfaz del Gemelo Digital — Central Geotérmica Double Flash 25 MWe (Proyecto 12, INQ-3108)
Ejecutar:   streamlit run app/app.py
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from simulador import planta as P
from simulador import termo
from pinn import modelo as M

st.set_page_config(page_title="Gemelo Digital Geotérmico", page_icon="🌋", layout="wide")
UMBRAL_DEF, EWMA, PEN_P1 = 0.08, 0.2, 0.05
COL_DATOS = P.COLS_MEDIDAS


@st.cache_resource
def cargar_modelo(caso):
    ruta = RAIZ / "pinn" / "modelos" / f"pinn_{caso}.pt"
    if not ruta.exists():
        return None
    return M.cargar(ruta)[0]


@st.cache_data
def simular_serie(caso, esc, semilla, nu):
    return P.serie_temporal(caso, esc, dias=180, seed=semilla, nu=nu)


@torch.no_grad()
def estimar(caso, red, df):
    med = M.a_tensores(df)
    _, _, eta, mh, fis = M.perdidas(caso, red, med, None, {})
    phi = ((1 - eta / fis["eta_limpia"]) / P.PAR["b_foul"]).numpy()
    return phi, eta.numpy(), fis["W_neta"].numpy(), fis


def ewma(x, a=EWMA):
    e, out = x[0], []
    for v in x:
        e = (1 - a) * e + a * v
        out.append(e)
    return np.array(out)


st.title("🌋 Gemelo Digital Termodinámico — Central Geotérmica Double Flash 25 MWe")
st.caption("Universidad Autónoma de Santo Domingo · Termodinámica II (INQ-3108) · Proyecto 12 · Datos simulados (con extensión a datos reales)")

with st.sidebar:
    st.header("Configuración")
    caso = st.radio("Caso de fluido", ["A", "B"], format_func=lambda c: "A · Agua + NCG (CO₂)" if c == "A" else "B · Salmuera NaCl + CO₂/H₂S")
    esc = st.selectbox("Escenario de operación", ["normal", "rapida", "pico_ncg"],
                       format_func=lambda e: {"normal": "Normal (ensuciamiento gradual)", "rapida": "Ensuciamiento rápido (×3)",
                                              "pico_ncg": "Pico de gases no condensables (días 90-100)"}[e])
    nu = st.slider("Nivel de ruido de los sensores (× base)", 0.5, 10.0, 1.0, 0.5)
    semilla = st.number_input("Semilla aleatoria", 0, 999, 1)
    umbral = st.slider("Umbral de alarma (φ estimado)", 0.03, 0.30, UMBRAL_DEF, 0.01)

red = cargar_modelo(caso)
if red is None:
    st.error("No se encontró el modelo entrenado. Ejecuta primero:  python run_all.py")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Gemelo en operación", "🎛️ Simulación manual", "🧪 Termodinámica (Cap. 12)",
                                        "⚖️ Estudio 2ª Ley", "📂 Datos reales"])

# ------------------------------------------------------------------------------------------------
with tab1:
    df = simular_serie(caso, esc, int(semilla), nu)
    phi_hat, eta_hat, W_fis, fis = estimar(caso, red, df)
    sesgo_base = float(np.mean(phi_hat[:7]))        # calibración de línea base: primeros 7 días tras limpieza
    ew = ewma(phi_hat - sesgo_base)
    idx = np.where(ew > umbral)[0]
    d_alarma = int(idx[0]) if len(idx) else None
    ip = np.where(df["P1"].values / P.PAR["P1"] - 1 > PEN_P1)[0]
    d_pen = int(ip[0]) if len(ip) else None

    dia = st.slider("Día de operación (reproducción)", 0, len(df) - 1, min(len(df) - 1, 60))
    sub = df.iloc[:dia + 1]
    en_alarma = d_alarma is not None and dia >= d_alarma
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado del gemelo", "🔴 ALERTA: ensuciamiento" if en_alarma else "🟢 Normal")
    c2.metric("Potencia neta medida", f"{df['W_neta_med'][dia]/1000:.2f} MW",
              f"{(df['W_neta_med'][dia]-df['W_neta_limpia'][dia])/1000:+.2f} MW vs planta limpia")
    c3.metric("φ estimado (media móvil)", f"{ew[dia]:.3f}", f"real: {df['phi'][dia]:.3f}", delta_color="off")
    c4.metric("Anticipación de la alarma",
              f"{d_pen - d_alarma} días" if (d_alarma is not None and d_pen is not None) else "—",
              "antes de +5 % en P₁" if d_pen is not None else None, delta_color="off")

    g1 = go.Figure()
    g1.add_scatter(x=sub.dia, y=sub.W_neta_med / 1000, name="Potencia medida (con ruido)", mode="markers", marker=dict(size=4, opacity=0.5))
    g1.add_scatter(x=sub.dia, y=sub.W_neta_limpia / 1000, name="Esperada con turbina limpia (gemelo)", line=dict(dash="dash"))
    g1.add_scatter(x=sub.dia, y=pd.Series(W_fis[:dia + 1]) / 1000, name="Potencia física con η estimado")
    g1.update_layout(title="Potencia neta [MW]", xaxis_title="día", yaxis_title="MW", height=330, legend=dict(orientation="h"))
    g2 = go.Figure()
    g2.add_scatter(x=sub.dia, y=sub.phi, name="φ real (simulador)", line=dict(color="black"))
    g2.add_scatter(x=sub.dia, y=phi_hat[:dia + 1], name="φ estimado (PINN)", mode="markers", marker=dict(size=4, opacity=0.4))
    g2.add_scatter(x=sub.dia, y=ew[:dia + 1], name="φ estimado (media móvil)", line=dict(color="crimson"))
    g2.add_hline(y=umbral, line_dash="dot", annotation_text="umbral de alarma")
    if d_alarma is not None and dia >= d_alarma:
        g2.add_vline(x=d_alarma, line_color="crimson", annotation_text="alarma")
    if d_pen is not None and dia >= d_pen:
        g2.add_vline(x=d_pen, line_color="gray", annotation_text="+5 % P₁")
    g2.update_layout(title="Índice de ensuciamiento φ (0 = limpia)", xaxis_title="día", height=330, legend=dict(orientation="h"))
    ca, cb = st.columns(2)
    ca.plotly_chart(g1, width="stretch")
    cb.plotly_chart(g2, width="stretch")

    g3 = go.Figure()
    g3.add_scatter(x=sub.dia, y=sub.P1_med, name="P₁ medida [bar]", mode="markers", marker=dict(size=4, opacity=0.5))
    g3.add_scatter(x=sub.dia, y=sub.P1, name="P₁ real", line=dict(color="black"))
    g3.add_hline(y=P.PAR["P1"] * (1 + PEN_P1), line_dash="dot", annotation_text="penalización +5 %")
    g3.update_layout(title="Presión del primer flash / entrada a turbina", xaxis_title="día", height=300, legend=dict(orientation="h"))
    g4 = go.Figure()
    S_gen = (fis["m_hp"] * fis["sig1"] + fis["m_lp"] * fis["sig2"]).numpy()[:dia + 1]
    g4.add_scatter(x=sub.dia, y=S_gen, name="Generación de entropía estimada [kW/K]")
    g4.add_scatter(x=sub.dia, y=sub.S_gen_turb, name="Real (simulador)", line=dict(color="black"))
    g4.add_hline(y=0, line_color="red")
    g4.update_layout(title="Segunda Ley: σ̇ estimada ≥ 0", xaxis_title="día", height=300)
    cc, cd = st.columns(2)
    cc.plotly_chart(g3, width="stretch")
    cd.plotly_chart(g4, width="stretch")
    st.caption(f"Línea base calibrada con los primeros 7 días (sesgo restado: {sesgo_base:+.3f}). σ̇ = Σ ṁ·(s_salida − s_entrada) de los dos tramos de turbina con la η̂ de la red; debe ser siempre positiva (Segunda Ley).")

# ------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("Simulación manual de la central")
    c1, c2, c3, c4, c5 = st.columns(5)
    m = c1.slider("Flujo de salmuera [kg/s]", 250.0, 380.0, 320.0)
    T = c2.slider("Temperatura [°C]", 215.0, 240.0, 230.0)
    y = c3.slider("NCG en vapor [% peso]", 1.0, 6.0, 2.8) / 100
    Tcw = c4.slider("Agua de enfriamiento [°C]", 20.0, 35.0, 28.0)
    phi = c5.slider("Ensuciamiento φ", 0.0, 0.8, 0.0, 0.01)
    B, ag = P.NumpyB, P.Agua(P.NumpyB)
    a = lambda v: np.array([v], float)
    filas = []
    for cs in ("A", "B"):
        r = P.planta(B, ag, cs, a(m), a(T), a(y), a(Tcw), a(phi))
        ssi = P.sobresaturacion_silice(r, a(T), P.PAR)[0]
        filas.append({"Caso": cs, "Potencia neta [MW]": r["W_neta"][0] / 1000, "Potencia bruta [MW]": r["W_bruta"][0] / 1000,
                      "Vapor 1er flash [kg/s]": r["m_hp"][0], "Vapor 2º flash [kg/s]": r["m_lp"][0],
                      "P₁ [bar]": r["P1"][0], "P₂ [bar]": r["P2"][0], "P condensador [bar]": r["Pc"][0],
                      "η efectiva": r["eta_ef"][0], "σ̇ turbina [kW/K]": r["S_gen_turb"][0],
                      "Exergía destruida turbina [MW]": r["Ex_dest_turb"][0] / 1000, "SSI sílice": ssi,
                      "dφ/dt [1/día]": float(P.dphi_dt(ssi, phi, P.PAR))})
    st.dataframe(pd.DataFrame(filas).set_index("Caso").T.style.format("{:.4f}"), width="stretch")
    st.caption("A = agua pura + CO₂ como NCG · B = salmuera 3 % NaCl (elevación del punto de ebullición, Raoult) + NCG con 10 % H₂S. "
               "SSI > 1: la sílice amorfa está sobresaturada y puede precipitar.")

# ------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("Relaciones del Capítulo 12 verificadas con IAPWS-95")
    tv = pd.DataFrame(termo.tabla_verificacion(), columns=["Relación", "Lado izquierdo", "Lado derecho", "Error relativo [%]"])
    st.dataframe(tv.style.format({"Lado izquierdo": "{:.6e}", "Lado derecho": "{:.6e}", "Error relativo [%]": "{:.2e}"}), width="stretch")
    Pn = st.slider("Presión para Newton-Raphson [bar]", 0.5, 25.0, 6.5)
    Ts, it = termo.temperatura_saturacion_newton(Pn)
    st.write(f"Newton-Raphson: Tsat({Pn:.2f} bar) = **{Ts:.3f} °C** en {it} iteraciones.")

# ------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("¿Qué pesos de penalización garantizan la Segunda Ley frente al ruido de sensores?")
    ruta = RAIZ / "resultados" / "estudio_segunda_ley.csv"
    if ruta.exists():
        e = pd.read_csv(ruta)
        reg = st.radio("Régimen", ["U", "S"], format_func=lambda r: "U · sin etiquetas (física + datos, como con datos reales)" if r == "U" else "S · supervisado con datos simulados", horizontal=True)
        tab = e[e.regimen == reg].groupby(["config", "nu"]).agg(violaciones_pct=("viol_pct", "mean"), mae_phi=("mae_phi", "mean"),
                                                                  sigma_min=("sigma_min", "min")).reset_index()
        piv = tab.pivot(index="config", columns="nu", values="violaciones_pct")
        st.write("**% de predicciones con σ < 0** (columnas: nivel de ruido ν × base, con 3 % de picos de sensor)")
        st.dataframe(piv.style.format("{:.2f}"), width="stretch")
        piv2 = tab.pivot(index="config", columns="nu", values="mae_phi")
        st.write("**Error absoluto medio de φ**")
        st.dataframe(piv2.style.format("{:.3f}"), width="stretch")
    else:
        st.info("Ejecuta `python run_all.py` para generar el estudio.")

# ------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("Extensión a datos reales")
    st.write("Sube un CSV con una fila por registro (p. ej. promedio diario) y estas columnas:")
    plantilla = pd.DataFrame({"m_brine": [320.0], "T_brine": [230.0], "y_ncg": [0.028], "Tcw": [28.0], "P1": [6.5], "P2": [1.8],
                              "Pc": [0.139], "W_neta": [25000.0], "m_vapor": [71.0]})
    st.dataframe(plantilla, width="stretch")
    st.caption("Unidades: flujos kg/s · temperaturas °C · y_ncg fracción en peso (0.028 = 2.8 %) · presiones bar · W_neta kW.")
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False), "plantilla_datos_reales.csv")
    up = st.file_uploader("CSV de datos de planta", type="csv")
    if up is not None:
        d = pd.read_csv(up)
        faltan = [c for c in COL_DATOS if c not in d.columns]
        if faltan:
            st.error(f"Faltan columnas: {faltan}")
        else:
            d2 = d.rename(columns={c: f"{c}_med" for c in COL_DATOS})
            ph, et, Wf, f = estimar(caso, red, d2)
            d["phi_estimado"], d["eta_estimada"], d["W_fisica_kW"] = ph, et, Wf
            st.dataframe(d.head(50), width="stretch")
            st.line_chart(d["phi_estimado"])
            st.warning("El modelo fue entrenado con datos simulados. Con datos reales conviene reentrenar/calibrar "
                       "(régimen U: sin etiquetas, solo balances físicos) antes de confiar en φ estimado.")
