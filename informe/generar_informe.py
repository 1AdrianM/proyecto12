"""
Genera el reporte PDF del Proyecto 12 a partir de los resultados reales (resultados/*.json|csv).
Uso: python informe/generar_informe.py      ->  informe/Proyecto12_Reporte.pdf
"""
import sys, json
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from simulador import planta as P, termo, separador as SEP
from pinn import modelo as M

RES, FIG = RAIZ / "resultados", RAIZ / "informe" / "fig"
FIG.mkdir(exist_ok=True)
met = json.load(open(RES / "metricas.json"))
est = pd.read_csv(RES / "estudio_segunda_ley.csv")

# ------------------------------------------------------------------ fuentes con soporte Unicode
d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for n, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
    pdfmetrics.registerFont(TTFont(n, str(d / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

AZ = colors.HexColor("#1f3a5f")
S = dict(
    body=ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5),
    h1=ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=18, textColor=AZ, spaceBefore=10, spaceAfter=6),
    h2=ParagraphStyle("h2", fontName="DV-B", fontSize=11, leading=14, textColor=AZ, spaceBefore=7, spaceAfter=4),
    eq=ParagraphStyle("eq", fontName="DV-I", fontSize=9.5, leading=14, alignment=TA_CENTER, spaceAfter=4, spaceBefore=2),
    cap=ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#444444")),
    cell=ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=9.6),
    cellb=ParagraphStyle("cellb", fontName="DV-B", fontSize=7.8, leading=9.6, textColor=colors.white),
    tit=ParagraphStyle("tit", fontName="DV-B", fontSize=17, leading=22, alignment=TA_CENTER, textColor=AZ, spaceAfter=8),
    sub=ParagraphStyle("sub", fontName="DV", fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=4),
    nota=ParagraphStyle("nota", fontName="DV", fontSize=8.3, leading=11.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#333333"), leftIndent=10, rightIndent=10, spaceAfter=6),
)
E = []
p = lambda t, s="body": E.append(Paragraph(t, S[s]))
h1 = lambda t: E.append(Paragraph(t, S["h1"]))
h2 = lambda t: E.append(Paragraph(t, S["h2"]))
eq = lambda t: E.append(Paragraph(t, S["eq"]))


def tabla(filas, anchos=None, cab=True):
    datos = [[Paragraph(str(c), S["cellb"] if (i == 0 and cab) else S["cell"]) for c in f] for i, f in enumerate(filas)]
    t = Table(datos, colWidths=anchos, repeatRows=1 if cab else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")])]
    if cab:
        st.append(("BACKGROUND", (0, 0), (-1, 0), AZ))
    t.setStyle(TableStyle(st))
    E.append(KeepTogether([t]))
    E.append(Spacer(1, 6))


def figura(nombre, ancho_cm, pie):
    E.append(KeepTogether([Image(str(FIG / nombre), width=ancho_cm * cm, height=ancho_cm * cm * Image(str(FIG / nombre)).imageHeight / Image(str(FIG / nombre)).imageWidth), Paragraph(pie, S["cap"])]))


f2 = lambda x, n=2: f"{x:.{n}f}"
B, ag = P.NumpyB, P.Agua(P.NumpyB)
a = lambda v: np.array([v], float)
nom = {c: P.planta(B, ag, c, a(320), a(230), a(0.028), a(28), a(0.0)) for c in "AB"}
ssi_nom = {c: float(P.sobresaturacion_silice(nom[c], a(230), P.PAR)[0]) for c in "AB"}

# =============================================================================================== figuras
# F1 arquitectura
fig, ax = plt.subplots(figsize=(8.6, 2.6)); ax.axis("off"); ax.grid(False)
cajas = [(0.0, "Planta real\no simulador\n(sensores IoT)"), (2.2, "Ingesta y\nlimpieza de datos"), (4.4, "Gemelo digital\nPINN + balances\n(1ª y 2ª Ley)"),
         (6.6, "Diagnóstico:\nη̂, φ̂, σ̇, exergía"), (8.8, "Alarma de\nensuciamiento")]
for x, t in cajas:
    ax.add_patch(plt.Rectangle((x, 0.55), 1.9, 1.1, fc="#dbe7f5", ec="#1f3a5f", lw=1.2)); ax.text(x + 0.95, 1.1, t, ha="center", va="center", fontsize=8)
for x in (1.9, 4.1, 6.3, 8.5):
    ax.annotate("", xy=(x + 0.3, 1.1), xytext=(x, 1.1), arrowprops=dict(arrowstyle="->", lw=1.3))
ax.add_patch(plt.Rectangle((3.3, -0.35), 3.4, 0.6, fc="#fde9d9", ec="#c0504d", lw=1.2)); ax.text(5.0, -0.05, "Histórico → reentrenamiento periódico", ha="center", va="center", fontsize=8)
ax.annotate("", xy=(5.3, 0.55), xytext=(5.3, 0.25), arrowprops=dict(arrowstyle="<-", lw=1.1, color="#c0504d"))
ax.set_xlim(-0.1, 10.9); ax.set_ylim(-0.5, 1.9)
fig.tight_layout(); fig.savefig(FIG / "f1_arquitectura.png"); plt.close(fig)

# F2 potencia y eficiencia vs phi
phis = np.linspace(0, 0.6, 31)
fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.8))
for c, col in (("A", "#1f77b4"), ("B", "#d62728")):
    r = P.planta(B, ag, c, np.full(31, 320.0), np.full(31, 230.0), np.full(31, 0.028), np.full(31, 28.0), phis)
    axs[0].plot(phis, r["W_neta"] / 1000, label=f"Caso {c}", color=col)
    axs[1].plot(phis, r["eta_ef"], color=col)
    axs[2].plot(phis, (P.PAR["T0"] + 273.15) * r["S_gen_turb"] / 1000, color=col)
axs[0].set(xlabel="φ", ylabel="Potencia neta [MW]", title="Potencia neta"); axs[0].legend()
axs[1].set(xlabel="φ", ylabel="η efectiva", title="Eficiencia isentrópica")
axs[2].set(xlabel="φ", ylabel="[MW]", title="Exergía destruida en turbina")
fig.tight_layout(); fig.savefig(FIG / "f2_phi.png"); plt.close(fig)

# F3 solubilidad de sílice
T = np.linspace(80, 250, 100)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(T, P.solubilidad_silice_amorfa(T), label="Sílice amorfa (Fournier & Rowe)")
ax.plot(T, 10 ** (-1309 / (T + 273.15) + 5.19), "--", label="Cuarzo (Fournier, 1977)")
r = nom["A"]
C_res = float(P.silice_reservorio(230.0, P.PAR))
C1 = C_res / (1 - float(r["x1"][0])); C2 = C_res / (float(r["m_liq2"][0]) / 320.0)
ax.plot([230, float(r["T1f"][0]), float(r["T2f"][0])], [C_res, C1, C2], "ko-", ms=5, label="Concentración en el líquido (reservorio → 1er flash → 2º flash)")
ax.set(xlabel="T [°C]", ylabel="SiO₂ [mg/kg]", title="Sobresaturación de sílice: SSI > 1 tras el 2º flash"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "f3_silice.png"); plt.close(fig)

# F4 series (caso A)
fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=False)
for ax_, esc, tit in zip(axs, ("normal", "rapida", "pico_ncg"), ("Normal", "Ensuciamiento rápido (×3)", "Pico de NCG (días 90-100)")):
    s = pd.read_csv(RES / f"serie_A_{esc}.csv")
    info = met[f"serie_A_{esc}"]
    ax_.plot(s.dia, s.phi, "k", label="φ real"); ax_.plot(s.dia, s.phi_hat, ".", ms=2, alpha=0.4, label="φ̂ diario")
    ax_.plot(s.dia, s.phi_hat_ewma, "r", label="φ̂ media móvil")
    ax_.axhline(0.08, ls=":", c="gray")
    if info["dia_alarma"] is not None: ax_.axvline(info["dia_alarma"], c="r", ls="--", lw=0.8)
    if info["dia_penalizacion"] is not None: ax_.axvline(info["dia_penalizacion"], c="gray", ls="--", lw=0.8)
    ax_.set(title=tit, xlabel="día"); ax_.set_ylim(-0.1, 0.9)
axs[0].set_ylabel("φ"); axs[0].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "f4_series.png"); plt.close(fig)

# F5 Lapple
sep = {k: SEP.lapple(P_, m_) for k, (P_, m_) in {"HP": (6.5, float(nom["A"]["m_hp"][0])), "LP": (1.8, float(nom["A"]["m_lp"][0]))}.items()}
dd = np.logspace(0, 2, 100)
fig, ax = plt.subplots(figsize=(4.8, 2.8))
for k, v in sep.items():
    ax.semilogx(dd, 100 * SEP.eficiencia_fraccional(dd, v["d50_um"]), label=f"Separador {k}: d50 = {v['d50_um']:.1f} µm")
ax.set(xlabel="diámetro de gota [µm]", ylabel="captura [%]", title="Eficiencia fraccional (Lapple)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "f5_lapple.png"); plt.close(fig)

# F6 estudio 2a ley
u = est[est.regimen == "U"]
g = u.groupby(["config", "nu"]).agg(v=("viol_pct", "mean"), s=("sigma_min", "min")).reset_index()
orden = ["soft lamS=0.0", "soft lamS=0.01", "soft lamS=0.1", "soft lamS=1.0", "soft lamS=10.0", "soft lamS=100.0"]
lams = [0, 0.01, 0.1, 1, 10, 100]
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.0))
for nu in (1.0, 5.0, 10.0):
    sub = g[g.nu == nu].set_index("config").loc[orden]
    axs[0].plot(range(6), sub.v, "o-", label=f"ruido ν = {nu:g}×")
    axs[1].semilogy(range(6), -sub.s, "o-")
for ax_ in axs: ax_.set_xticks(range(6)); ax_.set_xticklabels([str(l) for l in lams]); ax_.set_xlabel("peso λ_S de la 2ª Ley")
hard0 = g[g.config.str.startswith("dura")].v.max()
axs[0].axhline(hard0, c="g", ls="--", label="variante dura (sigmoide)"); axs[0].set_ylabel("% de predicciones con σ < 0"); axs[0].legend(fontsize=7)
axs[1].set_ylabel("|σ_mín| violada [kJ/(kg·K)]")
fig.tight_layout(); fig.savefig(FIG / "f6_segunda_ley.png"); plt.close(fig)

# F7 dispersión φ̂ vs φ (modelo final, caso A y B)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax_, c in zip(axs, "AB"):
    red, _ = M.cargar(RAIZ / "pinn" / "modelos" / f"pinn_{c}.pt")
    te = P.generar_dataset(c, 3000, seed=4242, nu=1.0)
    o, pr = M.evaluar(c, red, te)
    ax_.plot([0, 0.6], [0, 0.6], "k--", lw=0.8); ax_.plot(te.phi_true, pr["phi_hat"], ".", ms=2, alpha=0.4)
    ax_.set(xlabel="φ real", ylabel="φ estimado", title=f"Caso {c} (ruido base, MAE = {o['mae_phi']:.3f})")
fig.tight_layout(); fig.savefig(FIG / "f7_dispersion.png"); plt.close(fig)

# =============================================================================================== texto
pr_A, pr_B = met["precision_A"]["nu1"], met["precision_B"]["nu1"]
det_A = {r["escenario"]: r for r in met["deteccion_A"]}
det_B = {r["escenario"]: r for r in met["deteccion_B"]}
eco_A = met["economia_A"]
eco_base = [e for e in eco_A if e["precio"] == 80.0 and e["c_limpieza"] == 30000.0][0]
eco_max = max(eco_A, key=lambda e: e["ahorro_vs_calendario"])
t_inf = met["precision_A"]["t_inferencia_red_fisica_ms"]
ok = lambda c: "Cumple" if c else "No cumple"

# ---- portada
E.append(Spacer(1, 2.2 * cm))
p("UNIVERSIDAD AUTÓNOMA DE SANTO DOMINGO", "sub")
p("Facultad de Ingeniería y Arquitectura · Escuela de Ingeniería Química", "sub")
p("Asignatura: Termodinámica II (INQ-3108) · Semestre 2026-02", "sub")
E.append(Spacer(1, 1.2 * cm))
p("Proyecto 12", "tit")
p("Gemelo Digital (Digital Twin) Termodinámico con Redes Neuronales Informadas por la Física (PINNs) para Diagnóstico en Tiempo Real de una Central Geotérmica Flash", "tit")
E.append(Spacer(1, 1.0 * cm))
p("Central de doble evaporación flash de 25 MWe · Casos A (agua + NCG) y B (salmuera + CO₂/H₂S)", "sub")
E.append(Spacer(1, 2.0 * cm))
p("Integrantes: ______________________________  ·  ______________________________", "sub")
p("Profesor: Ing. Domingo Amparo", "sub")
p("Santo Domingo, diciembre de 2026", "sub")
E.append(Spacer(1, 1.2 * cm))
p("<b>Alcance.</b> Todo el análisis usa datos simulados (la planta virtual del Capítulo 3); no hay datos de una central real. "
  "Los resultados demuestran la metodología y su comportamiento bajo ruido, y el código admite datos reales (Capítulo 9).", "nota")
E.append(PageBreak())

# ---- 1 resumen
h1("1. Resumen")
p("Se desarrolló un gemelo digital de una central geotérmica de doble evaporación flash de 25 MWe. El gemelo combina (i) un modelo termodinámico basado en IAPWS-95 y en las relaciones "
  "del Capítulo 12 de Çengel &amp; Boles (Tds, Maxwell, Clapeyron, Joule-Thomson) con balances de masa, energía y entropía, (ii) una cinética de precipitación de sílice amorfa "
  "que degrada la turbina con el tiempo, y (iii) una red neuronal informada por la física (PINN) que estima la eficiencia real de la turbina y el índice de ensuciamiento φ a partir de sensores ruidosos. "
  "El sistema se probó con dos fluidos: agua + gases no condensables (caso A) y salmuera con NaCl y CO₂/H₂S (caso B).")
p(f"Con la planta limpia el modelo entrega {f2(nom['A']['W_neta'][0]/1000)} MWe netos (caso A, calibrado) y {f2(nom['B']['W_neta'][0]/1000)} MWe (caso B). "
  f"Con el nivel de ruido base de los sensores, la PINN estima la entalpía de escape con {f2(pr_A['err_rel_h_pct'],3)} % de error medio y el flujo de vapor con {f2(pr_A['err_rel_m_pct'],3)} % (caso A); "
  f"la inferencia toma {f2(t_inf,2)} ms. En los escenarios simulados de ensuciamiento, la alarma se activa en promedio {f2(det_A['normal']['anticip_media'],0)} días antes de que la presión de entrada "
  f"suba 5 % (escenario normal). La sección 7 responde la pregunta de decisión con un estudio numérico: una penalización blanda de la Segunda Ley reduce, pero no elimina, las violaciones; "
  "la garantía estricta se obtiene imponiendo la restricción por construcción.")

# ---- 2 requisitos
h1("2. Cumplimiento de las especificaciones del enunciado")
req = [["Requisito (horizonte 2026)", "Resultado en este trabajo", "Estado"],
       ["Potencia neta 25 MWe", f"A: {f2(nom['A']['W_neta'][0]/1000)} MWe (calibrado con η_seco = {P.PAR['eta_dry']:.3f}) · B: {f2(nom['B']['W_neta'][0]/1000)} MWe", f"Cumple (A); el caso B queda {100*(1-nom['B']['W_neta'][0]/nom['A']['W_neta'][0]):.1f} % por debajo por la salinidad"],
       ["Error de predicción < 0,75 % en entalpía y flujo", f"Caso A: entalpía {f2(pr_A['err_rel_h_pct'],3)} %, flujo {f2(pr_A['err_rel_m_pct'],3)} %  ·  Caso B: entalpía {f2(pr_B['err_rel_h_pct'],3)} %, flujo {f2(pr_B['err_rel_m_pct'],3)} % (frente a la planta simulada, ruido base)",
        ok(max(pr_A['err_rel_h_pct'], pr_A['err_rel_m_pct'], pr_B['err_rel_h_pct'], pr_B['err_rel_m_pct']) < 0.75) + " (solo contra datos simulados)"],
       ["Inferencia PINN < 85 ms", f"{f2(met['precision_A']['t_inferencia_red_ms'],3)} ms (solo red) · {f2(t_inf,3)} ms (red + capa física), CPU", ok(t_inf < 85)],
       ["Detección de incrustación > 14 días antes de la penalización de presión", f"Caso A normal: anticipación media {f2(det_A['normal']['anticip_media'],0)} d (mín. {f2(det_A['normal']['anticip_min'],0)} d); éxito ≥ 14 d: {f2(det_A['normal']['exito_14d'],0)} % de 20 simulaciones",
        ok(det_A['normal']['exito_14d'] >= 95) + " (en simulación)"],
       ["Ahorro operativo > 380 000 USD/año", f"Estimación base (80 USD/MWh, 30 000 USD/limpieza): {eco_base['ahorro_vs_calendario']:,.0f} USD/año vs. limpieza por calendario de 180 d; máximo en la sensibilidad: {eco_max['ahorro_vs_calendario']:,.0f}",
        ok(eco_base['ahorro_vs_calendario'] > 380000) + " en el caso base (ver Cap. 8)"]]
tabla(req, [4.6 * cm, 9.2 * cm, 3.9 * cm])

# ---- 3 modelo termodinámico
h1("3. Modelo termodinámico")
h2("3.1 Propiedades y relaciones del Capítulo 12")
p("Las propiedades del agua provienen de la ecuación de estado de Helmholtz IAPWS-95 (Wagner &amp; Pruß, 2002) a través de CoolProp. Como la entalpía y la entropía son derivadas de una "
  "función potencial, las relaciones de Maxwell se cumplen por construcción. Se verificaron numéricamente las siguientes relaciones:")
eq("dh = T ds + v dP    ·    (∂s/∂P)<sub>T</sub> = −(∂v/∂T)<sub>P</sub>    ·    dP<sub>sat</sub>/dT = h<sub>fg</sub> / (T v<sub>fg</sub>)    ·    μ<sub>JT</sub> = [T(∂v/∂T)<sub>P</sub> − v] / c<sub>p</sub>")
tv = termo.tabla_verificacion()
tabla([["Relación verificada (IAPWS-95)", "Lado izquierdo", "Lado derecho", "Error rel. [%]"]] +
      [[n, f"{x:.5e}", f"{y:.5e}", f"{e:.1e}"] for n, x, y, e in tv], [8.2 * cm, 3.0 * cm, 3.0 * cm, 2.4 * cm])
Ts_, it_ = termo.temperatura_saturacion_newton(6.5)
p(f"El solver Newton-Raphson (metodología computacional de Çengel &amp; Boles) resuelve P<sub>sat</sub>(T) − P = 0: para 6,5 bar obtiene T<sub>sat</sub> = {Ts_:.3f} °C en {it_} iteraciones.")

h2("3.2 Balances de la planta (régimen cuasi-estacionario)")
p("La salmuera entra como líquido saturado a 230 °C (28 bar) y se evapora en dos etapas: primer flash a P₁ = 6,5 bar y segundo flash a P₂ = 1,8 bar. Para cada flash:")
eq("ṁ<sub>in</sub> h<sub>in</sub> = ṁ<sub>v</sub> h<sub>v</sub> + (ṁ<sub>in</sub> − ṁ<sub>v</sub>) h<sub>l</sub>      →      caso A: x = (h<sub>in</sub> − h<sub>f</sub>) / h<sub>fg</sub>")
p("<b>Caso B (salmuera).</b> La sal permanece en el líquido, por lo que se concentra tras cada flash. La actividad del agua se aproxima con la ley de Raoult (NaCl con factor de van't Hoff 2), "
  "T<sub>flash</sub> = T<sub>sat</sub>(P / a<sub>w</sub>), y la entalpía de la salmuera es h = (1−w) h<sub>f</sub>(T) + w c<sub>s</sub> T, sin entalpía de mezcla. Los gases no condensables se toman como CO₂ (caso A) "
  "o 90 % CO₂ + 10 % H₂S en peso (caso B), con 2,8 % en peso en el vapor; afectan la presión del condensador y el trabajo de extracción de gases.")
p("<b>Turbina.</b> Cada corriente de vapor saturado se expande hasta la presión del condensador P<sub>c</sub> con eficiencia isentrópica η = η<sub>seco</sub>[1 − α(1 − x̄)]·(1 − bφ) "
  "(corrección de Baumann por humedad y pérdida por ensuciamiento). Por la Segunda Ley, la generación de entropía por tramo es")
eq("σ<sub>k</sub> = s<sub>sal</sub> − s<sub>ent</sub> = (1 − η) Δh<sub>is,k</sub> / T<sub>c</sub> ≥ 0      (porque h<sub>fg</sub> = T s<sub>fg</sub> en la zona húmeda, ecuación Tds)")
_t = termo.cargar_tablas()
_res = np.max(np.abs((_t["hg"] - _t["hf"]) - (_t["T"] + 273.15) * (_t["sg"] - _t["sf"])) / (_t["hg"] - _t["hf"]))
p("lo que equivale a η ≤ 1; la exergía destruida es Ex<sub>dest</sub> = T₀ · Σ ṁ<sub>k</sub> σ<sub>k</sub>. El código calcula σ con las tablas de s<sub>f</sub> y s<sub>fg</sub> y no con esta fórmula cerrada, de modo que la identidad Tds sirve como verificación de las tablas "
  f"(residuo relativo máximo de h<sub>fg</sub> − T s<sub>fg</sub> en las tablas IAPWS-95 usadas: {_res:.1e}).")
p("<b>Condensador barométrico (contacto directo).</b> Con ṁ<sub>cw</sub> = 1 900 kg/s de agua a T<sub>cw</sub> y una aproximación ΔT<sub>app</sub>: T<sub>c</sub> = T<sub>cw</sub> + Q̇/(ṁ<sub>cw</sub> c<sub>p</sub>) + ΔT<sub>app</sub>(1 + f<sub>cond</sub>), "
  "P<sub>c</sub> = P<sub>sat</sub>(T<sub>c</sub>) + k<sub>ej</sub> y<sub>NCG</sub>. El ensuciamiento del condensador entra como f<sub>cond</sub> (ΔT<sub>app</sub> = Q̇/(UaV), con 1/Ua = 1/Ua₀ + R<sub>f</sub>); en las simulaciones se dejó en 0 y queda como extensión.")

h2("3.3 Sistema de ecuaciones diferenciales algebraicas (DAE)")
p("El balance transitorio de la turbina geotérmica y del tren de eyección se formula como un sistema DAE de índice 1, con variables diferenciales (φ, temperatura de la masa metálica T<sub>t</sub>, inventario de NCG m<sub>NCG</sub>) y algebraicas (flujos, presiones, entalpías, potencias):")
eq("dφ/dt = k<sub>si</sub> · máx(SSI − 1, 0) · (1 − φ)")
eq("M<sub>t</sub>c<sub>t</sub> dT<sub>t</sub>/dt = Σ<sub>k</sub> ṁ<sub>k</sub>(h<sub>ent,k</sub> − h<sub>sal,k</sub>) − Ẇ − Q̇<sub>pérd</sub>          (1ª Ley, con acumulación)")
eq("dS<sub>t</sub>/dt = Σ<sub>k</sub> ṁ<sub>k</sub>(s<sub>ent,k</sub> − s<sub>sal,k</sub>) + Q̇/T + Ṡ<sub>gen</sub>,     Ṡ<sub>gen</sub> = Σ<sub>k</sub> ṁ<sub>k</sub> σ<sub>k</sub> ≥ 0          (2ª Ley)")
eq("dm<sub>NCG</sub>/dt = ṁ<sub>NCG,ent</sub> − ṁ<sub>NCG,ej</sub>,     0 = P<sub>c</sub> − P<sub>sat</sub>(T<sub>c</sub>) − k<sub>ej</sub> y<sub>NCG</sub>")
eq("0 = ṁ<sub>in</sub>h<sub>in</sub> − ṁ<sub>v</sub>h<sub>v</sub> − ṁ<sub>l</sub>h<sub>l</sub>,     0 = Ẇ<sub>k</sub> − ṁ<sub>k</sub> η<sub>k</sub> Δh<sub>is,k</sub>(P<sub>k</sub>, P<sub>c</sub>)")
p("<b>Simplificación adoptada.</b> La inercia térmica de la turbina y el inventario de NCG evolucionan en horas, mientras que el ensuciamiento lo hace en días. Por eso la implementación resuelve las ecuaciones algebraicas en régimen "
  "cuasi-estacionario (términos de acumulación = 0) y integra solo la ecuación diferencial de φ con paso diario. No se resolvió el DAE completo con acumulación.")

h2("3.4 Dimensionamiento preliminar del separador ciclónico")
tabla([["Separador", "P [bar]", "ṁ vapor [kg/s]", "Q [m³/s]", "D ciclón [m]", "d50 [µm]"]] +
      [[k, f2(v["P_bar"], 1), f2(v["m_vapor"], 1), f2(v["Q_m3s"], 1), f2(v["D_m"], 2), f2(v["d50_um"], 1)] for k, v in sep.items()],
      [3 * cm, 2.2 * cm, 3 * cm, 2.6 * cm, 2.9 * cm, 2.4 * cm])
p("Modelo de Lapple con velocidad de entrada de 20 m/s y 5 vueltas efectivas. Las gotas mayores a unas decenas de micras se capturan casi por completo; las gotas finas "
  "pueden arrastrar sílice disuelta hacia la turbina. Este cálculo es de diseño preliminar y no está acoplado al arrastre en el ensuciamiento simulado.")
figura("f5_lapple.png", 8.0, "Figura 1. Eficiencia fraccional de los separadores ciclónicos (modelo de Lapple).")

# ---- 4 resultados de diseño
h1("4. Resultados del modelo de la planta")
rA, rB = nom["A"], nom["B"]
fila = lambda n, k, fmt=lambda v: f2(v, 2), esc=1.0: [n, fmt(float(rA[k][0]) / esc), fmt(float(rB[k][0]) / esc)]
tabla([["Variable (planta limpia, 320 kg/s, 230 °C)", "Caso A", "Caso B"],
       fila("Vapor 1er flash [kg/s]", "m_hp"), fila("Vapor 2º flash [kg/s]", "m_lp"),
       fila("T 1er flash [°C]", "T1f"), fila("T 2º flash [°C]", "T2f"),
       fila("Presión del condensador [bar]", "Pc", lambda v: f2(v, 4)),
       fila("η efectiva de la turbina", "eta_ef", lambda v: f2(v, 4)),
       fila("Potencia bruta [MW]", "W_bruta", esc=1000), fila("Auxiliares [MW]", "W_aux", esc=1000), fila("Potencia neta [MW]", "W_neta", esc=1000),
       fila("σ̇ turbina [kW/K]", "S_gen_turb"), fila("Exergía destruida en turbina [MW]", "Ex_dest_turb", esc=1000),
       ["Sobresaturación de sílice SSI tras 2º flash", f2(ssi_nom["A"], 3), f2(ssi_nom["B"], 3)]], [8.2 * cm, 3.4 * cm, 3.4 * cm])
p("La salinidad del caso B eleva el punto de ebullición y reduce el vapor producido; con la misma turbina, la potencia neta es menor. Los resultados dependen de supuestos de ingeniería (eficiencias, auxiliares, "
  "condensador) listados en <i>simulador/planta.py</i>; η<sub>seco</sub> se calibró para que el caso A limpio dé 25 MWe.")
figura("f2_phi.png", 16.5, "Figura 2. Efecto del ensuciamiento φ sobre potencia neta, eficiencia y exergía destruida en la turbina.")

h2("4.1 Cinética de ensuciamiento por sílice")
p("La solubilidad de sílice amorfa sigue log₁₀ C = −731/T + 4,52 (T en K; Fournier &amp; Rowe, 1977) y la sílice del reservorio se toma en equilibrio con cuarzo, log₁₀ C = −1309/T + 5,19 (geotermómetro de cuarzo sin pérdida de vapor, Fournier, 1977). "
  f"Tras el segundo flash el líquido se concentra y se enfría, y el índice SSI = C/C<sub>sol</sub> pasa de 1 (A: {f2(ssi_nom['A'],3)}; B: {f2(ssi_nom['B'],3)}). El crecimiento del depósito se modela como "
  "dφ/dt = k<sub>si</sub>·(SSI − 1)·(1 − φ), con k<sub>si</sub> = 0,0262 d⁻¹ (valor de calibración elegido para que la penalización de +5 % en P₁ ocurra a los ~130 días con condiciones nominales; <b>no</b> proviene de mediciones). "
  "El ensuciamiento eleva P₁ en 15 % × φ y reduce η en 20 % × φ (coeficientes supuestos).")
figura("f3_silice.png", 9.5, "Figura 3. Solubilidad de sílice y concentración del líquido a lo largo de los flash (caso A).")

# ---- 5 PINN
h1("5. La PINN del gemelo digital")
p("La red (3 capas ocultas de 64 neuronas, SiLU) recibe nueve mediciones ruidosas: flujo y temperatura de salmuera, fracción de NCG, temperatura del agua de enfriamiento, P₁, P₂, P<sub>c</sub>, potencia neta y flujo de vapor. "
  "Entrega la eficiencia efectiva η̂ y el flujo de vapor ṁ̂. Una capa física diferenciable (las mismas ecuaciones del simulador, escritas en PyTorch y comprobadas contra NumPy con diferencia máxima de 0,03 kW en 25 000 kW) "
  "calcula la potencia, el balance de masa y σ. La función de pérdida es")
eq("L = λ<sub>D</sub> L<sub>datos</sub> + λ<sub>E</sub> L<sub>energía</sub> + λ<sub>M</sub> L<sub>masa</sub> + λ<sub>S</sub> L<sub>2ª Ley</sub>")
p("L<sub>energía</sub> compara la potencia neta física con la medida; L<sub>masa</sub> compara ṁ̂ con el balance de los flash; L<sub>2ª Ley</sub> = ⟨máx(−σ₁,0)² + máx(−σ₂,0)²⟩ + ⟨máx(−η̂,0)²⟩ se evalúa también en "
  "puntos de colocación sin etiqueta con ruido extremo y picos de sensor (el mecanismo clásico de una PINN). L<sub>datos</sub> supervisa η̂ y ṁ̂ con los valores verdaderos del simulador (solo posible con datos simulados). "
  "El índice de ensuciamiento se obtiene de φ̂ = (1 − η̂/η<sub>limpia</sub>)/b, con η<sub>limpia</sub> la eficiencia que tendría la turbina limpia en esas condiciones (línea base del gemelo). "
  "Las relaciones de Maxwell y Clapeyron actúan a través de las propiedades (tablas IAPWS-95 y la identidad h<sub>fg</sub> = T s<sub>fg</sub>) y no como términos separados de la pérdida; esa es una decisión de diseño, no una omisión.")
p(f"<b>Modelo final.</b> Se entrenó con η̂ restringida a (0,1) mediante una sigmoide (variante \"dura\", ver Cap. 7), λ<sub>D</sub> = λ<sub>E</sub> = λ<sub>M</sub> = 1 y λ<sub>S</sub> = {met['lam_S']:g}, "
  f"{met['epocas']} épocas, 20 000 muestras con ruido ν = 1,5× y 2 % de picos. Precisión con el ruido base (5 000 muestras nuevas): caso A, RMSE de η = {pr_A['rmse_eta']:.4f} y error absoluto medio de φ = {pr_A['mae_phi']:.3f}; "
  f"caso B, {pr_B['rmse_eta']:.4f} y {pr_B['mae_phi']:.3f}.")
figura("f1_arquitectura.png", 16.0, "Figura 4. Arquitectura del gemelo digital: del sensor a la alarma, con reentrenamiento periódico.")
figura("f7_dispersion.png", 12.0, "Figura 5. φ estimado frente a φ real con el ruido base (conjunto de prueba nuevo).")

# ---- 6 detección
h1("6. Detección temprana de ensuciamiento")
p("Se simularon 180 días de operación (condiciones variables, ruido de sensores ν = 1) en tres escenarios, con 20 semillas cada uno y para cada caso. La alarma salta cuando la media móvil exponencial (factor 0,2) de φ̂ supera 0,08, después de restar el sesgo medido en los primeros 7 días tras la puesta en marcha o la última limpieza (calibración de línea base, ver 6.1). "
  "La \"penalización de presión\" se define como un aumento de P₁ mayor a 5 % respecto al diseño. Una falsa alarma es una alarma con φ real < 0,03.")
fil = [["Caso", "Escenario", "Alarmas / n", "Anticipación media [d]", "Anticipación mín. [d]", "Éxito ≥ 14 d [%]", "Falsas alarmas", "MAE φ̂"]]
nombres = {"normal": "Normal", "rapida": "Rápido (×3)", "pico_ncg": "Pico de NCG"}
for c, det in (("A", det_A), ("B", det_B)):
    for esc in ("normal", "rapida", "pico_ncg"):
        r = det[esc]
        fil.append([c, nombres[esc], f"{r['alarmas']}/{r['n']}", f2(r["anticip_media"], 1), f2(r["anticip_min"], 0), f2(r["exito_14d"], 0) + (f" ({int(r['sin_pen'])} sin penalización)" if r.get("sin_pen", 0) else ""), int(r["falsas"]), f2(r["mae_phi"], 3)])
tabla(fil, [1.3 * cm, 2.4 * cm, 2.0 * cm, 2.6 * cm, 2.4 * cm, 3.0 * cm, 1.9 * cm, 1.6 * cm])
p("En el escenario rápido la penalización de P₁ llega antes y la anticipación en días es menor aunque φ̂ siga bien; el requisito de 14 días debe leerse junto con la velocidad de ensuciamiento. "
  "Donde dice \"sin penalización\", esa simulación no llegó al +5 % de P₁ en 180 días y se excluyó del cálculo de anticipación. "
  f"En el escenario con pico de NCG (5 % entre los días 90 y 100) el sesgo adicional de φ̂ durante el pico fue de {det_A['pico_ncg']['sesgo_pico']:+.3f} (caso A) y {det_B['pico_ncg']['sesgo_pico']:+.3f} (caso B), "
  "pequeño frente al error típico de φ̂ (≈ 0,03): la línea base conoce el cambio de gases por la medición de y<sub>NCG</sub> y no lo confunde con ensuciamiento. "
  "Como la alarma por ensuciamiento ya había saltado antes de los días 90-100, ese escenario no mide falsas alarmas; solo mide el sesgo.")
h2("6.1 Robustez: falsas alarmas y desajuste entre planta y modelo")
p("Como la PINN usa las mismas ecuaciones que el simulador, un error bajo frente a la planta simulada es optimista. Para medir qué pasa cuando la planta \"real\" difiere del modelo, se generaron datos con parámetros cambiados "
  "que la PINN no conoce, y se simularon 20 series de 180 días <b>sin ensuciamiento</b> (k<sub>si</sub> = 0) para contar falsas alarmas, con y sin la calibración de línea base.")
rb = met["robustez_A"]
tabla([["Diferencia entre planta y modelo (caso A)", "Sesgo de φ̂", "Falsas alarmas sin calibrar", "Falsas alarmas con calibración", "Anticipación con calibración [d]"]] +
      [[r["prueba"], f"{r['sesgo_phi']:+.3f}", f"{r['falsas_sin_cal']}/{r['n']}", f"{r['falsas_con_cal']}/{r['n']}", f2(r["anticip_media_cal"], 0)] for r in rb],
      [6.2 * cm, 2.2 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm])
p("Los cambios que los sensores sí ven (presión del condensador, gases) casi no afectan a φ̂. En cambio, una turbina que de fábrica rinde 2 % menos que el modelo se lee como φ ≈ 0,08 y, sin calibrar, "
  "dispara la alarma en todas las simulaciones con la planta limpia. Restar el sesgo de los primeros 7 días elimina esas falsas alarmas y conserva una anticipación de unos 100 días. "
  "Por eso el gemelo necesita un periodo de referencia con la turbina limpia antes de vigilar ensuciamiento, y la calibración debe repetirse después de cada limpieza.")
figura("f4_series.png", 17.0, "Figura 6. Series de ejemplo (caso A): φ real, φ̂ diario y media móvil, umbral (punteado), alarma (rojo) y penalización de P₁ (gris).")

# ---- 7 pregunta de decisión
h1("7. Pregunta de decisión: pesos de la 2ª Ley frente al ruido de sensores")
p("<i>¿Qué pesos de penalización en la función de pérdida multiobjetivo de la red PINN garantizan que las predicciones cumplan estrictamente el principio de incremento de entropía frente a ruido aleatorio de instrumentación en presión y temperatura?</i>")
p("Se entrenaron redes (caso A, 2 semillas) variando λ<sub>S</sub> ∈ {0; 0,01; 0,1; 1; 10; 100} y una variante \"dura\" con η̂ = sigmoide(·), y se evaluaron con ruido ν ∈ {0,5; 1; 2; 5; 10}× el base más 3 % de picos de sensor (factor 0,6 a 1,6). "
  "Régimen U (sin etiquetas): solo balances físicos y datos medidos, como ocurriría con datos reales. Régimen S: supervisado con datos simulados.")
fil = [["Configuración (régimen U)", "% violaciones ν=1", "% violaciones ν=10", "|σ_mín| violada ν=1 [kJ/kgK]", "MAE φ ν=1"]]
gu = u.groupby(["config", "nu"]).agg(v=("viol_pct", "mean"), s=("sigma_min", "min"), m=("mae_phi", "mean")).reset_index()
for cfg in orden + ["dura (sigmoide)"]:
    a1, a10 = gu[(gu.config == cfg) & (gu.nu == 1.0)].iloc[0], gu[(gu.config == cfg) & (gu.nu == 10.0)].iloc[0]
    fil.append([cfg.replace("soft lamS=", "λ_S = ").replace("dura (sigmoide)", "Dura (sigmoide)"), f2(a1.v, 2), f2(a10.v, 2), f"{max(-a1.s, 0):.4f}", f2(a1.m, 3)])
tabla(fil, [5.0 * cm, 2.8 * cm, 2.8 * cm, 3.7 * cm, 2.2 * cm])
fig_s = None
figura("f6_segunda_ley.png", 16.5, "Figura 7. Régimen U: porcentaje de predicciones con σ < 0 y magnitud de la violación según el peso λ_S.")
p("<b>Conclusiones del estudio.</b>")
p(f"1) Una penalización blanda <b>no garantiza</b> σ ≥ 0. Al subir λ<sub>S</sub> de 0 a 100 la magnitud de la violación cae de ≈{max(-gu[(gu.config=='soft lamS=0.0')&(gu.nu==1.0)].s.iloc[0],0):.2f} a "
  f"≈{max(-gu[(gu.config=='soft lamS=100.0')&(gu.nu==1.0)].s.iloc[0],0):.3f} kJ/(kg·K), pero con un penalizador cuadrático el equilibrio entre el término de energía (que empuja η̂ a seguir un pico del sensor) y el de la 2ª Ley "
  "queda siempre ligeramente del lado prohibido; la proporción de predicciones con σ < 0 baja a la mitad o menos, pero no a cero.")
p("2) La garantía estricta se logra imponiendo η̂ ∈ (0,1) por construcción (σ ≥ 0 equivale a η ≤ 1): 0 violaciones en todos los niveles de ruido, sin costo apreciable en el error de φ̂ (tabla anterior).")
_sv = est[est.regimen == "S"].groupby(["config", "nu"]).viol_pct.mean()
_cfg, _nu = _sv.idxmax()
p(f"3) Con supervisión (régimen S) casi no hay violaciones con ninguna configuración: la tasa máxima fue {_sv.max():.2f} % ({_cfg}, ν = {_nu:g}×); el resto dio 0,00 % salvo valores menores. "
  "Las etiquetas mantienen η̂ lejos del límite. El riesgo aparece cuando se entrena sin etiquetas, como sería con datos reales.")
_ms = est[est.regimen == "S"].groupby(["config", "nu"]).mae_phi.mean()
p("4) Los pesos deben escalarse con la incertidumbre del sensor: cada residuo está normalizado por su escala (1 % de la potencia nominal para la energía; 0,05 kJ/(kg·K) para σ). "
  f"<b>Hallazgo importante:</b> en el régimen supervisado, el MLP sin física estimó φ mejor que la PINN en todos los niveles de ruido (MAE con ν = 1: {_ms[('MLP sin física', 1.0)]:.3f} frente a {_ms[('PINN dura', 1.0)]:.3f}; "
  f"con ν = 10: {_ms[('MLP sin física', 10.0)]:.3f} frente a {_ms[('PINN dura', 10.0)]:.3f}). El término de energía con peso fijo obliga a la red a seguir la potencia medida, que trae ruido. "
  "La ventaja de la PINN en este trabajo no es la precisión con datos etiquetados, sino que puede entrenarse sin etiquetas (régimen U, el caso de datos reales) y que respeta la física. "
  "Para mejorar la precisión, λ<sub>E</sub> debería disminuir con σ<sub>sensor</sub>² (tabla completa en <i>resultados/estudio_segunda_ley.csv</i>).")
p("<b>Respuesta.</b> Ningún valor finito de λ<sub>S</sub> garantiza σ ≥ 0 de forma estricta en este estudio. Recomendación: restringir η̂ ∈ (0,1) por construcción y conservar un λ<sub>S</sub> ≥ 10 como penalización auxiliar. "
  "Esta conclusión es de un experimento con datos simulados y un solo tipo de falla de sensor (picos multiplicativos), no una demostración general.")

# ---- 8 economía
h1("8. Estimación económica")
p("Se compara el costo anual (pérdida de producción por ensuciamiento + limpiezas) de tres políticas: limpieza por calendario cada 180 días, limpieza reactiva cuando P₁ sube 5 % y limpieza basada en condición en el intervalo óptimo que el gemelo permite identificar. "
  "Cada limpieza cuesta C<sub>limpieza</sub> más 2 días de parada (producción perdida). La pérdida de potencia se toma de la curva simulada a condiciones nominales.")
fil = [["Precio [USD/MWh]", "C limpieza [USD]", "T calendario [d]", "T reactiva [d]", "T óptimo [d]", "φ en T óptimo", "Ahorro vs. calendario [USD/año]", "Ahorro vs. reactiva [USD/año]"]]
for e in eco_A:
    fil.append([f"{e['precio']:.0f}", f"{e['c_limpieza']:,.0f}", e["T_calendario"], e["T_reactiva"], e["T_optimo"], f2(e["phi_en_T_opt"], 3), f"{e['ahorro_vs_calendario']:,.0f}", f"{e['ahorro_vs_reactiva']:,.0f}"])
tabla(fil, [2.1 * cm, 2.2 * cm, 2.1 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm, 3.1 * cm, 2.9 * cm])
p(f"En el caso base el ahorro frente al calendario es de {eco_base['ahorro_vs_calendario']:,.0f} USD/año; frente a limpiar de forma reactiva es mucho menor ({eco_base['ahorro_vs_reactiva']:,.0f} USD/año), porque el intervalo reactivo ya está cerca del óptimo. "
  f"La cifra del enunciado (380 000 USD/año) {'se alcanza' if eco_base['ahorro_vs_calendario']>380000 else 'no se alcanza'} en el caso base; el resultado depende fuertemente del precio de la energía, del costo de limpieza y de la política de referencia, "
  "y no incluye paradas no programadas ni daños mecánicos evitados, que son el argumento económico habitual de estos sistemas.")

# ---- 9 datos reales / limitaciones
h1("9. Extensión a datos reales y limitaciones")
p("<b>Cómo se conectan datos reales.</b> En la interfaz (pestaña \"Datos reales\") se carga un CSV con las nueve columnas medidas (plantilla descargable). El modelo entrenado con simulación sirve de arranque; con datos reales debe reentrenarse en el régimen sin etiquetas "
  "(λ<sub>D</sub> = 0), solo con los balances físicos, y recalibrar los parámetros supuestos (eficiencias, auxiliares, aproximación del condensador, k<sub>si</sub>). Hace falta un periodo de referencia con la turbina limpia para fijar la línea base.")
p("<b>Limitaciones.</b> (1) No hay datos de planta; todo error reportado es frente a la planta simulada con el mismo modelo físico que usa la PINN, por lo que es optimista. (2) El fluido real agua-NaCl-CO₂-H₂S es más complejo que el modelo (Raoult sin entalpía de mezcla; NCG sin equilibrio líquido-vapor). "
  "(3) Los coeficientes de ensuciamiento (k<sub>si</sub>, a₁, a₂, b) son supuestos calibrados para el ejercicio. (4) El separador ciclónico se dimensionó pero no está acoplado al arrastre de sílice. (5) El DAE se resolvió en régimen cuasi-estacionario. "
  "(6) El ahorro depende de supuestos económicos explícitos. (7) Con datos etiquetados, una red sin física fue más precisa que la PINN (Cap. 7). (8) La detección exige calibrar la línea base tras cada limpieza (sección 6.1).")

# ---- 10 conclusiones
h1("10. Conclusiones")
p("El gemelo digital reproduce la central en los dos casos de fluido, estima el ensuciamiento por sílice a partir de sensores ruidosos y lo detecta con semanas de anticipación en simulación. "
  "La parte termodinámica (relaciones del Capítulo 12, balances de masa, energía y entropía, exergía) es la base del modelo, y la red neuronal la usa como restricción. "
  "En cuanto a la Segunda Ley, la garantía estricta requiere imponerla por construcción; la penalización por sí sola solo reduce las violaciones. El paso siguiente es validar con datos reales de una planta.")

h1("Referencias")
for r in ["Çengel, Y. A., &amp; Boles, M. A. <i>Termodinámica</i>, 7.ª ed. McGraw-Hill. Cap. 12 (relaciones de propiedades) y metodología computacional.",
          "Wagner, W., &amp; Pruß, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance. <i>J. Phys. Chem. Ref. Data</i>, 31, 387-535.",
          "Bell, I. H., Wronski, J., Quoilin, S., &amp; Lemort, V. (2014). Pure and pseudo-pure fluid thermophysical property evaluation and the open-source library CoolProp. <i>Ind. Eng. Chem. Res.</i>, 53, 2498-2508.",
          "Raissi, M., Perdikaris, P., &amp; Karniadakis, G. E. (2019). Physics-informed neural networks. <i>J. Comput. Phys.</i>, 378, 686-707.",
          "Fournier, R. O., &amp; Rowe, J. J. (1977). The solubility of amorphous silica in water at high temperatures and high pressures. <i>American Mineralogist</i>, 62, 1052-1056.",
          "Fournier, R. O. (1977). Chemical geothermometers and mixing models for geothermal systems. <i>Geothermics</i>, 5, 41-50.",
          "DiPippo, R. <i>Geothermal Power Plants: Principles, Applications, Case Studies and Environmental Impact</i>. Butterworth-Heinemann.",
          "Lapple, C. E. (1951). Processes use many collector types. <i>Chemical Engineering</i>, 58(5), 144-151."]:
    p(r)
p("Las correlaciones de solubilidad de sílice y los datos de estas referencias se escribieron de memoria; conviene verificarlos en las fuentes originales antes de la entrega.", "nota")


def pie(canvas, doc):
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "Proyecto 12 · Gemelo digital termodinámico · INQ-3108 · Semestre 2026-02")
    canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Pág. {doc.page}")


doc = SimpleDocTemplate(str(RAIZ / "informe" / "Proyecto12_Reporte.pdf"), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Proyecto 12 - Gemelo Digital Termodinámico con PINNs")
doc.build(E, onFirstPage=pie, onLaterPages=pie)
print("PDF generado:", RAIZ / "informe" / "Proyecto12_Reporte.pdf")
