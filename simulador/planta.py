"""
Simulador de la central geotérmica de doble evaporación flash (Double Flash), 25 MWe.

Dos casos de fluido:
  Caso A  "agua + NCG":   agua pura (IAPWS-95) + gases no condensables tratados como CO2.
  Caso B  "salmuera":     salmuera con NaCl (elevación del punto de ebullición por la ley de Raoult),
                          NCG = 90 % CO2 + 10 % H2S en peso (cambia la masa molar y el trabajo de
                          extracción de gases).

El código está escrito de forma "agnóstica del backend": las mismas ecuaciones se ejecutan con
NumPy (simulador) y con PyTorch (capa física de la PINN, derivable por autodiferenciación).

Unidades: P [bar], T [°C], h [kJ/kg], s [kJ/(kg K)], flujos [kg/s], potencias [kW] salvo indicación.
"""
import numpy as np
from .termo import cargar_tablas

# ----------------------------------------------------------------------------------------------
# Parámetros del problema (enunciado) y supuestos de ingeniería
# ----------------------------------------------------------------------------------------------
PAR = dict(
    # --- enunciado ---
    m_brine=320.0, T_brine=230.0, P_brine=28.0,   # salmuera: 320 kg/s a 230 °C y 28 bar
    P1=6.5, P2=1.8,                               # primer y segundo flash [bar]
    y_ncg=0.028,                                  # NCG en el vapor: 2.8 % en peso
    W_neta_diseno=25000.0,                        # kW
    # --- supuestos de ingeniería (documentados en el informe) ---
    eta_dry=0.7963,      # eficiencia isentrópica "en seco" (calibrada: 25.0 MWe netos, caso A, limpio)
    alpha_baumann=0.65,  # corrección por humedad: eta = eta_dry [1 - alpha (1 - x_prom)]
    b_foul=0.20,         # pérdida de eficiencia por ensuciamiento: eta = eta0 (1 - b*phi)
    a1_foul=0.15,        # aumento de presión 1er flash por ensuciamiento: P1 (1 + a1*phi)
    a2_foul=0.08,        # idem 2do flash
    eta_gen=0.97,        # eficiencia del generador
    cs_nacl=0.85,        # cp de la sal [kJ/kg K]
    cp_vapor=2.10,       # cp del vapor [kJ/kg K]
    cp_agua=4.18,
    w_nacl_B=0.03,       # fracción másica de NaCl (caso B)
    f_h2s=0.10,          # fracción másica de H2S dentro del NCG (caso B)
    m_cw=1900.0,         # caudal de agua de enfriamiento del condensador de contacto directo [kg/s]
    dT_app=3.0,          # aproximación del condensador [K]
    f_cond=0.0,          # ensuciamiento del condensador: dT_app_efectiva = dT_app (1 + f_cond)  (0 = limpio)
    k_ej=0.5,            # presión parcial residual de NCG en el condensador por unidad de y_ncg [bar]
    T0=25.0,             # temperatura ambiente de referencia exergética [°C]
    aux_frac=0.025,      # auxiliares proporcionales a la potencia bruta (bombas/ventiladores)
    aux_fijo=1000.0,     # auxiliares fijos (reinyección, control) [kW]
    eta_comp=0.60,       # eficiencia del sistema de extracción de gases
    P_atm=1.013,
    # --- cinética de sílice ---
    k_si=0.0262,         # [1/día] por unidad de exceso de saturación (SSI-1)
    f_si_res=1.0,        # factor sobre la sílice de reservorio (equilibrio con cuarzo)
)

CASOS = ("A", "B")


# ----------------------------------------------------------------------------------------------
# Backends (NumPy / PyTorch)
# ----------------------------------------------------------------------------------------------
class NumpyB:
    nombre = "numpy"
    log, exp = staticmethod(np.log), staticmethod(np.exp)
    relu = staticmethod(lambda x: np.maximum(x, 0.0))
    clip = staticmethod(np.clip)
    maximum = staticmethod(np.maximum)
    arr = staticmethod(lambda x: np.asarray(x, dtype=float))

    @staticmethod
    def interp(x, xs, ys):
        return np.interp(x, xs, ys)


def hacer_torch_backend(dtype=None):
    import torch
    dt = dtype or torch.float32

    class TorchB:
        nombre = "torch"
        log, exp = staticmethod(torch.log), staticmethod(torch.exp)
        relu = staticmethod(torch.relu)
        clip = staticmethod(torch.clamp)
        maximum = staticmethod(torch.maximum)
        arr = staticmethod(lambda x: torch.as_tensor(x, dtype=dt))

        @staticmethod
        def interp(x, xs, ys):
            idx = torch.clamp(torch.searchsorted(xs, x.detach().contiguous()), 1, len(xs) - 1)
            x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)

    TorchB.dtype = dt
    return TorchB


class Agua:
    """Propiedades de saturación del agua por interpolación en las tablas IAPWS-95."""

    def __init__(self, B):
        t = cargar_tablas()
        self.B = B
        conv = lambda a: B.arr(a)
        self.lnP = conv(np.log(t["P"]))
        self.T = conv(t["T"])
        self.tab = {k: conv(t[k]) for k in ("hf", "hg", "sf", "sg", "vf", "vg", "T")}
        self.tabT = {k: conv(t[k]) for k in ("hf", "hg", "sf", "sg")}

    def p(self, nombre, P_bar):          # propiedad en función de la presión
        return self.B.interp(self.B.log(P_bar), self.lnP, self.tab[nombre])

    def pT(self, nombre, T_C):           # propiedad en función de la temperatura de saturación
        return self.B.interp(T_C, self.T, self.tabT[nombre])

    def Psat(self, T_C):
        return self.B.exp(self.B.interp(T_C, self.T, self.lnP))

    def Tsat(self, P_bar):
        return self.p("T", P_bar)


# ----------------------------------------------------------------------------------------------
# Salmuera (caso B): elevación del punto de ebullición (Raoult) y entalpía
# ----------------------------------------------------------------------------------------------
def actividad_agua(w):
    """a_w ≈ fracción molar de agua con factor de van't Hoff i=2 para NaCl."""
    n_w = (1.0 - w) * 1000.0 / 18.015
    n_s = w * 1000.0 / 58.44 * 2.0
    return n_w / (n_w + n_s)


def entalpia_salmuera(ag, P_eq, w, par):
    """h [kJ/kg salmuera] = (1-w) hf(T) + w cs T  (sin entalpía de mezcla)."""
    T = ag.Tsat(P_eq)
    return (1.0 - w) * ag.p("hf", P_eq) + w * par["cs_nacl"] * T


def flash(B, ag, P, m_in, h_in, w_in, par, caso):
    """
    Evaporación instantánea (flash) a presión P.  Balance de masa y energía:
        m_in h_in = m_v h_v + (m_in - m_v) h_l(T_f, w_l)
    Caso A: w = 0  ->  x = (h_in - h_f)/h_fg  (fórmula clásica).
    Caso B: la sal queda en el líquido; Tf = Tsat(P / a_w) (Raoult) -> se itera w_l.
    Devuelve m_v, m_l, h_l, w_l, T_f.
    """
    if caso == "A":
        hf, hfg = ag.p("hf", P), ag.p("hg", P) - ag.p("hf", P)
        m_v = m_in * (h_in - hf) / hfg
        m_l = m_in - m_v
        return m_v, m_l, hf, w_in * 0.0 + 0.0, ag.Tsat(P)
    m_v = m_in * 0.1
    for _ in range(6):                                   # punto fijo (converge en pocas iteraciones)
        m_l = m_in - m_v
        w_l = w_in * m_in / m_l
        P_eq = P / actividad_agua(w_l)                   # presión de saturación del agua pura equivalente
        Tf = ag.Tsat(P_eq)
        h_l = entalpia_salmuera(ag, P_eq, w_l, par)
        h_v = ag.p("hg", P) + par["cp_vapor"] * (Tf - ag.Tsat(P))
        m_v = m_in * (h_in - h_l) / (h_v - h_l)
    m_l = m_in - m_v
    return m_v, m_l, h_l, w_in * m_in / m_l, Tf


# ----------------------------------------------------------------------------------------------
# Turbina (un tramo de expansión desde vapor saturado a P_in hasta la presión del condensador)
# ----------------------------------------------------------------------------------------------
def etapa_ref(B, ag, P_in, Pc):
    """Estado de entrada (vapor saturado) y expansión isentrópica hasta Pc (zona húmeda)."""
    h_in, s_in = ag.p("hg", P_in), ag.p("sg", P_in)
    hf, hg, sf, sg = ag.p("hf", Pc), ag.p("hg", Pc), ag.p("sf", Pc), ag.p("sg", Pc)
    x_is = (s_in - sf) / (sg - sf)
    h_is = hf + x_is * (hg - hf)
    return dict(h_in=h_in, s_in=s_in, x_is=x_is, h_is=h_is, dh_is=h_in - h_is, hf=hf, hfg=hg - hf, sf=sf, sfg=sg - sf)


def eta_limpia(par, x_is):
    """Eficiencia de una turbina limpia con corrección de Baumann (humedad media del tramo)."""
    x_prom = 0.5 * (1.0 + x_is)
    return par["eta_dry"] * (1.0 - par["alpha_baumann"] * (1.0 - x_prom))


def estado_salida(ref, eta):
    """h, x y s a la salida con eficiencia isentrópica 'eta' y generación de entropía específica."""
    h_out = ref["h_in"] - eta * ref["dh_is"]
    x_out = (h_out - ref["hf"]) / ref["hfg"]
    s_out = ref["sf"] + x_out * ref["sfg"]
    return h_out, x_out, s_out, s_out - ref["s_in"]


def masa_molar_ncg(par, caso):
    if caso == "A":
        return 44.01
    return 1.0 / ((1 - par["f_h2s"]) / 44.01 + par["f_h2s"] / 34.08)


def auxiliares(B, ag, W_bruta, m_agua, y, Pc, T_c, par, caso):
    """Auxiliares [kW]: fracción de la bruta + fijo + compresión de NCG de Pc a P_atm."""
    m_ncg = y / (1.0 - y) * m_agua
    R = 8.314 / masa_molar_ncg(par, caso)                      # kJ/(kg K)
    w_comp = R * (T_c + 273.15) * B.log(par["P_atm"] / Pc) / par["eta_comp"]
    return par["aux_frac"] * W_bruta + par["aux_fijo"] + m_ncg * w_comp


# ----------------------------------------------------------------------------------------------
# Modelo completo de la planta (gemelo "físico")
# ----------------------------------------------------------------------------------------------
def planta(B, ag, caso, m_brine, T_brine, y_ncg, Tcw, phi, par=None):
    """
    Resuelve la central para un estado de operación y un índice de ensuciamiento phi ∈ [0,1].
    Todas las entradas pueden ser arreglos (vectorizado).
    """
    par = par or PAR
    w = par["w_nacl_B"] if caso == "B" else 0.0
    # presiones de flash afectadas por el ensuciamiento (la turbina restringida eleva la presión aguas arriba)
    P1 = par["P1"] * (1.0 + par["a1_foul"] * phi)
    P2 = par["P2"] * (1.0 + par["a2_foul"] * phi)

    # entalpía de entrada (líquido saturado a T_brine)
    if caso == "A":
        h_in = ag.pT("hf", T_brine)
    else:
        h_in = (1.0 - w) * ag.pT("hf", T_brine) + w * par["cs_nacl"] * T_brine

    m_hp, m_l1, h_l1, w_l1, T1f = flash(B, ag, P1, m_brine, h_in, w, par, caso)
    m_lp, m_l2, h_l2, w_l2, T2f = flash(B, ag, P2, m_l1, h_l1, w_l1 if caso == "B" else w, par, caso)

    # condensador de contacto directo: iteración sobre Pc (amortiguada)
    Pc = 0.12 + 0.0 * m_brine
    for _ in range(25):
        r1, r2 = etapa_ref(B, ag, P1, Pc), etapa_ref(B, ag, P2, Pc)
        eta1 = eta_limpia(par, r1["x_is"]) * (1.0 - par["b_foul"] * phi)
        eta2 = eta_limpia(par, r2["x_is"]) * (1.0 - par["b_foul"] * phi)
        h1o, x1o, s1o, sig1 = estado_salida(r1, eta1)
        h2o, x2o, s2o, sig2 = estado_salida(r2, eta2)
        T_out = Tcw + 0.0 * Pc
        Q = m_hp * (h1o - ag.pT("hf", T_out)) + m_lp * (h2o - ag.pT("hf", T_out))
        for _ in range(3):                                   # T_out depende de Q
            T_out = Tcw + Q / (par["m_cw"] * par["cp_agua"])
            Q = m_hp * (h1o - ag.pT("hf", T_out)) + m_lp * (h2o - ag.pT("hf", T_out))
        T_c = T_out + par["dT_app"] * (1.0 + par["f_cond"])
        Pc_nuevo = ag.Psat(T_c) + par["k_ej"] * y_ncg
        Pc = 0.5 * Pc + 0.5 * Pc_nuevo

    W1, W2 = m_hp * eta1 * r1["dh_is"], m_lp * eta2 * r2["dh_is"]
    W_bruta = par["eta_gen"] * (W1 + W2)
    m_agua = m_hp + m_lp
    W_aux = auxiliares(B, ag, W_bruta, m_agua, y_ncg, Pc, T_c, par, caso)
    W_neta = W_bruta - W_aux
    dh_pond = (m_hp * r1["dh_is"] + m_lp * r2["dh_is"])
    eta_ef = (W1 + W2) / dh_pond                              # eficiencia efectiva ponderada
    S_gen = m_hp * sig1 + m_lp * sig2                         # kW/K, generación de entropía en turbina
    # irreversibilidad en los separadores flash (aprox. con propiedades del agua)
    S_flash = (m_hp * ag.p("sg", P1) + m_l1 * ag.p("sf", P1) - m_brine * ag.pT("sf", T_brine)) \
        + (m_lp * ag.p("sg", P2) + m_l2 * ag.p("sf", P2) - m_l1 * ag.p("sf", P1))
    if caso == "B":                      # con sal el cálculo con entropías del agua pura no es válido
        S_flash = S_flash * np.nan if B.nombre == "numpy" else S_flash
    return dict(P1=P1, P2=P2, Pc=Pc, T_c=T_c, m_hp=m_hp, m_lp=m_lp, m_vapor=m_agua, m_liq2=m_l2,
                x1=m_hp / m_brine, T1f=T1f, T2f=T2f, w_l2=w_l2,
                eta1=eta1, eta2=eta2, eta_ef=eta_ef, W_bruta=W_bruta, W_aux=W_aux, W_neta=W_neta,
                W_turb_isentropica=dh_pond, h_exh=(m_hp * h1o + m_lp * h2o) / m_agua, S_gen_turb=S_gen, S_gen_flash=S_flash,
                Ex_dest_turb=(par["T0"] + 273.15) * S_gen,
                Ex_dest_flash=(par["T0"] + 273.15) * S_flash, h_out1=h1o, h_out2=h2o, x_out1=x1o, x_out2=x2o)


def calibrar_eta_dry(caso="A", objetivo=None):
    """Ajusta eta_dry para que la potencia neta limpia sea la del enunciado (25 MWe). Bisección."""
    objetivo = objetivo or PAR["W_neta_diseno"]
    B, ag = NumpyB, Agua(NumpyB)
    lo, hi = 0.60, 1.00
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        par = dict(PAR, eta_dry=mid)
        r = planta(B, ag, caso, *[np.array([v]) for v in (PAR["m_brine"], PAR["T_brine"], PAR["y_ncg"], 28.0, 0.0)], par=par)
        if r["W_neta"][0] < objetivo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------------
# Cinética de ensuciamiento por sílice amorfa
# ----------------------------------------------------------------------------------------------
def solubilidad_silice_amorfa(T_C):
    """Fournier & Rowe (1977): log10 C[mg/kg] = -731/T[K] + 4.52."""
    return 10.0 ** (-731.0 / (T_C + 273.15) + 4.52)


def silice_reservorio(T_C, par):
    """Sílice en equilibrio con cuarzo (Fournier & Potter): log10 C = -1309/T + 5.19."""
    return par["f_si_res"] * 10.0 ** (-1309.0 / (T_C + 273.15) + 5.19)


def sobresaturacion_silice(estado, T_brine, par):
    """SSI = C_silice(líquido tras el 2º flash) / C_sol_amorfa(T_2º flash). SSI>1 => precipita."""
    C_res = silice_reservorio(T_brine, par)
    factor = 1.0 / (estado["m_liq2"] / estado_m_brine(estado))
    return (C_res * factor) / solubilidad_silice_amorfa(estado["T2f"])


def estado_m_brine(estado):
    return estado["m_hp"] / estado["x1"]


def dphi_dt(ssi, phi, par, mult=1.0):
    """d(phi)/dt [1/día]: crece con el exceso de sobresaturación y se frena al acercarse a phi=1."""
    return mult * par["k_si"] * np.maximum(ssi - 1.0, 0.0) * (1.0 - phi)


# ----------------------------------------------------------------------------------------------
# Sensores (ruido) y generación de datos
# ----------------------------------------------------------------------------------------------
SIGMA_BASE = dict(m_brine=0.005, T_brine=0.35, y_ncg=0.04, Tcw=0.30, P1=0.003, P2=0.003,
                  Pc=0.010, W_neta=0.004, m_vapor=0.008)     # relativos salvo T (absoluto, °C)
COLS_MEDIDAS = ["m_brine", "T_brine", "y_ncg", "Tcw", "P1", "P2", "Pc", "W_neta", "m_vapor"]


def medir(rng, verdad, nu=1.0, p_outlier=0.0, par=None):
    """Aplica ruido gaussiano (nivel nu × base) y, opcionalmente, 'picos' de sensor en W y P1."""
    out = {}
    for k in COLS_MEDIDAS:
        v = np.asarray(verdad[k], float)
        s = SIGMA_BASE[k] * nu
        out[k] = v + rng.normal(0, s, v.shape) if k in ("T_brine", "Tcw") else v * (1 + rng.normal(0, s, v.shape))
    if p_outlier > 0:
        for k in ("W_neta", "P1", "m_vapor"):
            m = rng.random(out[k].shape) < p_outlier
            out[k] = np.where(m, out[k] * rng.uniform(0.6, 1.6, out[k].shape), out[k])
    return out


def muestrear_operacion(rng, n, phi_max=0.6):
    return dict(m_brine=rng.uniform(290, 350, n), T_brine=rng.uniform(222, 236, n),
                y_ncg=rng.uniform(0.015, 0.040, n), Tcw=rng.uniform(24, 32, n),
                phi=rng.uniform(0.0, phi_max, n))


def generar_dataset(caso, n, seed=0, nu=1.0, p_outlier=0.0, par=None, phi_max=0.6):
    """Tabla de verdades (planta simulada) + mediciones ruidosas, lista para entrenar."""
    import pandas as pd
    par = par or PAR
    rng = np.random.default_rng(seed)
    B, ag = NumpyB, Agua(NumpyB)
    op = muestrear_operacion(rng, n, phi_max)
    r = planta(B, ag, caso, op["m_brine"], op["T_brine"], op["y_ncg"], op["Tcw"], op["phi"], par)
    verdad = dict(op, **{k: r[k] for k in ("P1", "P2", "Pc", "W_neta", "eta_ef", "m_vapor", "S_gen_turb", "W_bruta", "h_exh")})
    med = medir(rng, verdad, nu, p_outlier, par)
    df = pd.DataFrame({f"{k}_med": v for k, v in med.items()})
    for k in ("phi", "eta_ef", "W_neta", "m_vapor", "Pc", "S_gen_turb", "h_exh"):
        df[f"{k}_true"] = verdad[k]
    return df


def serie_temporal(caso, escenario="normal", dias=180, seed=1, nu=1.0, par=None, dt_limpieza=None):
    """
    Simula 'dias' días de operación con ensuciamiento progresivo.
    Escenarios: 'normal', 'rapida' (cinética x3), 'pico_ncg' (NCG sube a 5 % entre los días 90 y 100).
    Devuelve un DataFrame con verdad + mediciones ruidosas por día.
    """
    import pandas as pd
    par = par or PAR
    rng = np.random.default_rng(seed)
    B, ag = NumpyB, Agua(NumpyB)
    mult = 3.0 if escenario == "rapida" else 1.0
    m, T, y, Tc = 320.0, 230.0, 0.028, 28.0
    phi, filas = 0.0, []
    for d in range(dias):
        m = 320 + 0.9 * (m - 320) + rng.normal(0, 3.0)                 # variación lenta de operación (AR(1))
        T = 230 + 0.9 * (T - 230) + rng.normal(0, 0.6)
        y = 0.028 + 0.9 * (y - 0.028) + rng.normal(0, 0.0006)
        Tc = 28 + 0.85 * (Tc - 28) + rng.normal(0, 0.5)
        y_eff = 0.05 if (escenario == "pico_ncg" and 90 <= d < 100) else y
        arr = lambda v: np.array([v], float)
        r = planta(B, ag, caso, arr(m), arr(T), arr(y_eff), arr(Tc), arr(phi), par)
        ssi = sobresaturacion_silice(r, arr(T), par)[0]
        fila = dict(dia=d, m_brine=m, T_brine=T, y_ncg=y_eff, Tcw=Tc, phi=phi, SSI=ssi,
                    **{k: float(r[k][0]) for k in ("P1", "P2", "Pc", "W_neta", "eta_ef", "m_vapor", "S_gen_turb", "W_bruta", "h_exh")})
        filas.append(fila)
        phi = float(np.clip(phi + dphi_dt(ssi, phi, par, mult), 0, 1))
    df = pd.DataFrame(filas)
    med = medir(rng, {k: df[k].values for k in COLS_MEDIDAS}, nu, 0.0, par)
    for k, v in med.items():
        df[f"{k}_med"] = v
    # potencia que tendría la planta limpia en las mismas condiciones (línea base del gemelo)
    r0 = planta(B, ag, caso, df["m_brine"].values, df["T_brine"].values, df["y_ncg"].values, df["Tcw"].values, np.zeros(len(df)), par)
    df["W_neta_limpia"] = r0["W_neta"]
    df["eta_ef_limpia"] = r0["eta_ef"]
    return df


if __name__ == "__main__":
    ed = calibrar_eta_dry("A")
    print("eta_dry calibrada (caso A):", round(ed, 4))
