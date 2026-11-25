"""
Módulo termodinámico — Capítulo 12 de Çengel & Boles (relaciones de propiedades).

Qué hace este archivo:
  1. Genera tablas de saturación del agua con la ecuación IAPWS-95 (vía CoolProp).
  2. Verifica numéricamente las relaciones del capítulo 12:
        - ecuación Tds / dh = T ds + v dP
        - relaciones de Maxwell:  (∂s/∂P)_T = -(∂v/∂T)_P
        - ecuación de Clapeyron:  dPsat/dT = h_fg / (T v_fg)
        - coeficiente de Joule-Thomson:  μ = [T (∂v/∂T)_P - v] / c_p
  3. Incluye un solver Newton-Raphson genérico (metodología computacional de Çengel & Boles).

Unidades de las tablas: P [bar], T [°C], h [kJ/kg], s [kJ/(kg·K)], v [m³/kg].
"""
from pathlib import Path
import numpy as np
from CoolProp.CoolProp import PropsSI

AGUA = "Water"
RUTA_TABLAS = Path(__file__).with_name("tablas_agua.npz")


# ----------------------------------------------------------------------------
# 1) Tablas de saturación (IAPWS-95)
# ----------------------------------------------------------------------------
def construir_tablas(n=600, p_min_bar=0.02, p_max_bar=40.0, guardar=True):
    """Calcula propiedades de líquido y vapor saturados en una malla de presiones (log)."""
    P_bar = np.geomspace(p_min_bar, p_max_bar, n)
    P = P_bar * 1e5
    T = np.array([PropsSI("T", "P", p, "Q", 0, AGUA) for p in P]) - 273.15
    hf = np.array([PropsSI("H", "P", p, "Q", 0, AGUA) for p in P]) / 1e3
    hg = np.array([PropsSI("H", "P", p, "Q", 1, AGUA) for p in P]) / 1e3
    sf = np.array([PropsSI("S", "P", p, "Q", 0, AGUA) for p in P]) / 1e3
    sg = np.array([PropsSI("S", "P", p, "Q", 1, AGUA) for p in P]) / 1e3
    vf = 1.0 / np.array([PropsSI("D", "P", p, "Q", 0, AGUA) for p in P])
    vg = 1.0 / np.array([PropsSI("D", "P", p, "Q", 1, AGUA) for p in P])
    tablas = dict(P=P_bar, T=T, hf=hf, hg=hg, sf=sf, sg=sg, vf=vf, vg=vg)
    if guardar:
        np.savez(RUTA_TABLAS, **tablas)
    return tablas


def cargar_tablas():
    if not RUTA_TABLAS.exists():
        construir_tablas()
    d = np.load(RUTA_TABLAS)
    return {k: d[k] for k in d.files}


# ----------------------------------------------------------------------------
# 2) Verificaciones del capítulo 12
# ----------------------------------------------------------------------------
def _derivada_centrada(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def verificar_maxwell(T_C=200.0, P_bar=5.0):
    """(∂s/∂P)_T  contra  -(∂v/∂T)_P   (cuarta relación de Maxwell)."""
    T, P = T_C + 273.15, P_bar * 1e5
    s = lambda p: PropsSI("S", "T", T, "P", p, AGUA)
    v = lambda t: 1.0 / PropsSI("D", "T", t, "P", P, AGUA)
    lado_izq = _derivada_centrada(s, P, 50.0)          # J/(kg K Pa)
    lado_der = -_derivada_centrada(v, T, 0.01)          # m3/(kg K)
    return lado_izq, lado_der


def verificar_tds(T_C=200.0, P_bar=5.0):
    """(∂h/∂P)_T  contra  v - T (∂v/∂T)_P   (consecuencia de dh = T ds + v dP)."""
    T, P = T_C + 273.15, P_bar * 1e5
    h = lambda p: PropsSI("H", "T", T, "P", p, AGUA)
    v_P = 1.0 / PropsSI("D", "T", T, "P", P, AGUA)
    dvdT = _derivada_centrada(lambda t: 1.0 / PropsSI("D", "T", t, "P", P, AGUA), T, 0.01)
    return _derivada_centrada(h, P, 50.0), v_P - T * dvdT


def verificar_clapeyron(T_C=162.0):
    """dPsat/dT numérico  contra  h_fg / (T v_fg)."""
    T = T_C + 273.15
    psat = lambda t: PropsSI("P", "T", t, "Q", 1, AGUA)
    dpdt_num = _derivada_centrada(psat, T, 0.01)
    hfg = PropsSI("H", "T", T, "Q", 1, AGUA) - PropsSI("H", "T", T, "Q", 0, AGUA)
    vfg = 1.0 / PropsSI("D", "T", T, "Q", 1, AGUA) - 1.0 / PropsSI("D", "T", T, "Q", 0, AGUA)
    return dpdt_num, hfg / (T * vfg)


def joule_thomson(T_C=200.0, P_bar=5.0):
    """μ_JT calculado con la relación termodinámica y comparado con CoolProp [K/Pa]."""
    T, P = T_C + 273.15, P_bar * 1e5
    v = 1.0 / PropsSI("D", "T", T, "P", P, AGUA)
    dvdT = _derivada_centrada(lambda t: 1.0 / PropsSI("D", "T", t, "P", P, AGUA), T, 0.01)
    cp = PropsSI("C", "T", T, "P", P, AGUA)
    mu_rel = (T * dvdT - v) / cp
    mu_cp = PropsSI("d(T)/d(P)|H", "T", T, "P", P, AGUA)
    return mu_rel, mu_cp


def tabla_verificacion():
    """Devuelve una lista de filas (nombre, valor_relacion, valor_referencia, error_rel_%)."""
    filas = []
    a, b = verificar_maxwell(200, 5);    filas.append(("Maxwell: (ds/dP)_T vs -(dv/dT)_P  [vapor 200 °C, 5 bar]", a, b))
    a, b = verificar_maxwell(300, 20);   filas.append(("Maxwell: (ds/dP)_T vs -(dv/dT)_P  [vapor 300 °C, 20 bar]", a, b))
    a, b = verificar_tds(200, 5);        filas.append(("Tds: (dh/dP)_T vs v - T(dv/dT)_P  [vapor 200 °C, 5 bar]", a, b))
    a, b = verificar_clapeyron(162);     filas.append(("Clapeyron: dPsat/dT vs hfg/(T vfg)  [162 °C]", a, b))
    a, b = verificar_clapeyron(230);     filas.append(("Clapeyron: dPsat/dT vs hfg/(T vfg)  [230 °C]", a, b))
    a, b = joule_thomson(200, 5);        filas.append(("Joule-Thomson: [T(dv/dT)_P - v]/cp vs CoolProp  [200 °C, 5 bar]", a, b))
    a, b = joule_thomson(300, 20);       filas.append(("Joule-Thomson: [T(dv/dT)_P - v]/cp vs CoolProp  [300 °C, 20 bar]", a, b))
    return [(n, x, y, 100.0 * abs(x - y) / abs(y)) for n, x, y in filas]


# ----------------------------------------------------------------------------
# 3) Newton-Raphson genérico (Çengel & Boles, metodología computacional)
# ----------------------------------------------------------------------------
def newton_raphson(f, x0, tol=1e-10, max_iter=50, h=1e-6):
    """Resuelve f(x)=0 con derivada numérica. Devuelve (x, iteraciones)."""
    x = float(x0)
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = (f(x + h * max(1.0, abs(x))) - fx) / (h * max(1.0, abs(x)))
        paso = fx / dfx
        x -= paso
        if abs(paso) < tol * max(1.0, abs(x)):
            return x, i
    raise RuntimeError("Newton-Raphson no convergió")


def temperatura_saturacion_newton(P_bar, T0_C=100.0):
    """Ejemplo del método: encuentra Tsat(P) resolviendo Psat(T) - P = 0 con Newton-Raphson."""
    f = lambda t: PropsSI("P", "T", t + 273.15, "Q", 1, AGUA) / 1e5 - P_bar
    return newton_raphson(f, T0_C)


if __name__ == "__main__":
    construir_tablas()
    for n, a, b, e in tabla_verificacion():
        print(f"{n:78s} {a: .6e} {b: .6e}  err={e:.4f}%")
    print("Newton Tsat(6.5 bar) =", temperatura_saturacion_newton(6.5))
