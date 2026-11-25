"""
Entrenamiento final de las PINN (casos A y B) y evaluación del gemelo digital:
  - precisión (error de potencia, de entalpía de escape y de flujo de vapor frente a los "sensores")
  - tiempo de inferencia
  - detección temprana de ensuciamiento por sílice en series temporales (3 escenarios)
  - estimación económica con supuestos explícitos
Salidas: pinn/modelos/*.pt  y  resultados/*.json|csv
"""
import sys, json, time
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, torch
from simulador import planta as P
from pinn import modelo as M

torch.set_num_threads(2)
LAM_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0       # peso de la 2a Ley elegido tras el estudio
EPOCAS = 150
SOLO = len(sys.argv) > 3 and sys.argv[3] == "solo"
MODO = sys.argv[2] if len(sys.argv) > 2 else "hard"     # "hard": η ∈ (0,1) por construcción (garantiza σ >= 0)
RES = RAIZ / "resultados"
(RAIZ / "pinn" / "modelos").mkdir(exist_ok=True)
metricas = {"lam_S": LAM_S, "epocas": EPOCAS, "modo": MODO}

UMBRAL_PHI = 0.08          # umbral de alarma sobre la media móvil exponencial de phi_hat
EWMA = 0.2
PENALIZACION_P1 = 0.05     # "penalización de presión": P1 sube más de 5 % respecto al diseño


def entrenar_caso(caso):
    if SOLO:                                           # reutiliza el modelo ya entrenado
        return M.cargar(RAIZ / "pinn" / "modelos" / f"pinn_{caso}.pt")[0]
    tr = P.generar_dataset(caso, 20000, seed=1000, nu=1.5, p_outlier=0.02)
    col = P.generar_dataset(caso, 20000, seed=1001, nu=6.0, p_outlier=0.15)
    red = M.entrenar(caso, tr, col, MODO, dict(D=1.0, E=1.0, M=1.0, S=LAM_S), epocas=EPOCAS, seed=0)
    M.guardar(red, RAIZ / "pinn" / "modelos" / f"pinn_{caso}.pt", caso)
    return red


def precision(caso, red):
    out = {}
    for nombre, nu, po in (("nu1", 1.0, 0.0), ("nu2", 2.0, 0.0), ("nu5_picos", 5.0, 0.03)):
        te = P.generar_dataset(caso, 5000, seed=5000 + int(nu * 10), nu=nu, p_outlier=po)
        o, _ = M.evaluar(caso, red, te)
        out[nombre] = o
    # tiempo de inferencia (una muestra, solo red / red + capa física)
    te = P.generar_dataset(caso, 200, seed=77, nu=1.0)
    med = M.a_tensores(te)
    x1 = M.matriz({k: v[:1] for k, v in med.items()})
    mb1 = M.balance_masa(caso, {k: v[:1] for k, v in med.items()})
    with torch.no_grad():
        for _ in range(20):
            red(x1, mb1)
        t = time.perf_counter()
        for _ in range(500):
            red(x1, mb1)
        t_red = (time.perf_counter() - t) / 500 * 1e3
        t = time.perf_counter()
        for _ in range(200):
            m1 = {k: v[:1] for k, v in med.items()}
            e, mh = red(M.matriz(m1), M.balance_masa(caso, m1))
            M.capa_fisica(caso, m1, e, mh)
        t_tot = (time.perf_counter() - t) / 200 * 1e3
    out["t_inferencia_red_ms"], out["t_inferencia_red_fisica_ms"] = t_red, t_tot
    return out


@torch.no_grad()
def phi_diario(caso, red, df):
    med = M.a_tensores(df)
    _, _, eta, mh, fis = M.perdidas(caso, red, med, None, {})
    return ((1 - eta / fis["eta_limpia"]) / P.PAR["b_foul"]).numpy(), eta.numpy(), fis


DIAS_CAL = 7               # días de referencia tras puesta en marcha/limpieza para fijar la línea base


def alarma(phi_hat, calibrar=True):
    """Media móvil exponencial de φ̂ (restando el sesgo medido en los primeros DIAS_CAL días) y día de alarma."""
    if calibrar:
        phi_hat = phi_hat - float(np.mean(phi_hat[:DIAS_CAL]))
    e, ew = phi_hat[0], []
    for v in phi_hat:
        e = (1 - EWMA) * e + EWMA * v
        ew.append(e)
    ew = np.array(ew)
    idx = np.where(ew > UMBRAL_PHI)[0]
    return ew, (int(idx[0]) if len(idx) else None)


def detectar_series(caso, red, n_semillas=20):
    filas, ejemplo = [], {}
    for esc in ("normal", "rapida", "pico_ncg"):
        for s in range(n_semillas):
            df = P.serie_temporal(caso, esc, dias=180, seed=300 + s, nu=1.0)
            ph, eta, _ = phi_diario(caso, red, df)
            ew, d_alarma = alarma(ph)
            idx_pen = np.where(df["P1"].values / P.PAR["P1"] - 1 > PENALIZACION_P1)[0]
            d_pen = int(idx_pen[0]) if len(idx_pen) else None
            falsa = d_alarma is not None and df["phi"].values[d_alarma] < 0.03
            filas.append(dict(caso=caso, escenario=esc, semilla=s, dia_alarma=d_alarma, dia_penalizacion=d_pen,
                              anticipacion_dias=(d_pen - d_alarma) if (d_alarma is not None and d_pen is not None) else None,
                              falsa_alarma=bool(falsa), phi_en_alarma=float(df["phi"].values[d_alarma]) if d_alarma is not None else None,
                              mae_phi=float(np.abs(ph - df["phi"].values).mean()),
                              sesgo_pico=float((ph - df["phi"].values)[90:100].mean() - (ph - df["phi"].values)[80:90].mean())))
            if s == 0:
                df = df.assign(phi_hat=ph, phi_hat_ewma=ew, eta_hat=eta)
                ejemplo[esc] = dict(df=df, dia_alarma=d_alarma, dia_pen=d_pen)
    return pd.DataFrame(filas), ejemplo


def robustez(caso, red, n=20):
    """Falsas alarmas con planta sin ensuciamiento y efecto de desajustes entre la planta 'real' y la física de la PINN."""
    filas = []
    pruebas = [("Sin desajuste", {}), ("Condensador sucio (f_cond = 0,5)", dict(f_cond=0.5)),
               ("Auxiliares +20 %", dict(aux_frac=P.PAR["aux_frac"] * 1.2)),
               ("Turbina 2 % menos eficiente de fábrica", dict(eta_dry=P.PAR["eta_dry"] * 0.98)),
               ("Extracción de NCG peor (k_ej × 2)", dict(k_ej=P.PAR["k_ej"] * 2))]
    for nombre, cambio in pruebas:
        par = dict(P.PAR, **cambio)
        df = P.generar_dataset(caso, 4000, seed=77, nu=1.0, par=par)
        _, pr = M.evaluar(caso, red, df)
        sesgo = float(np.mean(pr["phi_hat"] - df["phi_true"]))
        falsas = {True: 0, False: 0}
        antic = []
        for s in range(n):
            limpia = P.serie_temporal(caso, "normal", 180, seed=300 + s, par=dict(par, k_si=0.0))
            ph, _, _ = phi_diario(caso, red, limpia)
            for cal in (True, False):
                falsas[cal] += alarma(ph, cal)[1] is not None
            sucia = P.serie_temporal(caso, "normal", 180, seed=300 + s, par=par)
            ph, _, _ = phi_diario(caso, red, sucia)
            d = alarma(ph, True)[1]
            ip = np.where(sucia["P1"].values / P.PAR["P1"] - 1 > PENALIZACION_P1)[0]
            if d is not None and len(ip):
                antic.append(int(ip[0]) - d)
        filas.append(dict(prueba=nombre, sesgo_phi=sesgo, mae_phi=float(np.abs(pr["phi_hat"] - df["phi_true"]).mean()),
                          falsas_sin_cal=falsas[False], falsas_con_cal=falsas[True], n=n,
                          anticip_media_cal=float(np.mean(antic)) if antic else None))
    return filas


def curva_perdida(caso, dias=500):
    """Pérdida de potencia neta [kW] y phi tras una limpieza, a condiciones nominales."""
    B, ag = P.NumpyB, P.Agua(P.NumpyB)
    a = lambda v: np.array([v], float)
    phi, t, ph, dW, W = 0.0, [], [], [], []
    W0 = P.planta(B, ag, caso, a(320), a(230), a(0.028), a(28), a(0.0))["W_neta"][0]
    for d in range(dias):
        r = P.planta(B, ag, caso, a(320), a(230), a(0.028), a(28), a(phi))
        ssi = P.sobresaturacion_silice(r, a(230), P.PAR)[0]
        t.append(d); ph.append(phi); W.append(r["W_neta"][0]); dW.append(W0 - r["W_neta"][0])
        phi = float(np.clip(phi + P.dphi_dt(ssi, phi, P.PAR), 0, 1))
        if False: pass
    return np.array(t), np.array(ph), np.array(dW), W0, np.array([P.planta(B, ag, caso, a(320), a(230), a(0.028), a(28), a(p))["P1"][0] for p in ph])


def economia(caso, precio=80.0, c_limpieza=30000.0, dias_parada=2.0):
    """Costo anual por ensuciamiento + limpiezas para tres políticas de limpieza (USD/año)."""
    t, ph, dW, W0, P1 = curva_perdida(caso)
    c_par = dias_parada * 24 * (W0 / 1000.0) * precio              # producción perdida en la parada
    C = c_limpieza + c_par
    perd_usd_dia = dW * 24 / 1000.0 * precio                        # USD/día por pérdida de potencia
    acum = np.cumsum(perd_usd_dia)

    def costo_anual(T):
        T = int(T)
        return (C + acum[T - 1]) / T * 365.0

    T_cal = 180
    idx = np.where(P1 / P.PAR["P1"] - 1 > PENALIZACION_P1)[0]
    T_reac = int(idx[0]) if len(idx) else 365
    Ts = np.arange(5, 400)
    costos = np.array([costo_anual(T) for T in Ts])
    T_opt = int(Ts[costos.argmin()])
    return dict(precio=precio, c_limpieza=c_limpieza, c_parada=c_par, T_calendario=T_cal, T_reactiva=T_reac, T_optimo=T_opt,
                costo_calendario=costo_anual(T_cal), costo_reactiva=costo_anual(T_reac), costo_optimo=float(costos.min()),
                ahorro_vs_calendario=costo_anual(T_cal) - float(costos.min()),
                ahorro_vs_reactiva=costo_anual(T_reac) - float(costos.min()),
                phi_en_T_opt=float(ph[T_opt - 1]), dP1_pct_en_T_opt=float(100 * (P1[T_opt - 1] / P.PAR["P1"] - 1)))


if __name__ == "__main__":
    t0 = time.time()
    ejemplos, todas = {}, []
    for caso in ("A", "B"):
        print(f"[{time.time()-t0:5.0f}s] entrenando caso {caso} (λ_S={LAM_S})", flush=True)
        red = entrenar_caso(caso)
        metricas[f"precision_{caso}"] = precision(caso, red)
        df_det, ej = detectar_series(caso, red)
        df_det.to_csv(RES / f"deteccion_{caso}.csv", index=False)
        for esc, v in ej.items():
            v["df"].to_csv(RES / f"serie_{caso}_{esc}.csv", index=False)
            metricas.setdefault(f"serie_{caso}_{esc}", dict(dia_alarma=v["dia_alarma"], dia_penalizacion=v["dia_pen"]))
        res = df_det.groupby("escenario").agg(
            alarmas=("dia_alarma", lambda s: int(s.notna().sum())), n=("semilla", "size"),
            anticip_media=("anticipacion_dias", "mean"), anticip_min=("anticipacion_dias", "min"),
            exito_14d=("anticipacion_dias", lambda s: float((s.dropna() >= 14).mean() * 100)),
            sin_pen=("dia_penalizacion", lambda s: int(s.isna().sum())),
            falsas=("falsa_alarma", "sum"), mae_phi=("mae_phi", "mean"), sesgo_pico=("sesgo_pico", "mean"),
            phi_alarma=("phi_en_alarma", "mean"), dia_alarma=("dia_alarma", "mean"), dia_pen=("dia_penalizacion", "mean"))
        metricas[f"deteccion_{caso}"] = res.reset_index().to_dict(orient="records")
        metricas[f"economia_{caso}"] = [economia(caso, pr, cl) for pr in (60.0, 80.0, 100.0) for cl in (30000.0, 60000.0)]
        print(res.round(2), flush=True)
    metricas["robustez_A"] = robustez("A", M.cargar(RAIZ / "pinn" / "modelos" / "pinn_A.pt")[0])
    json.dump(metricas, open(RES / "metricas.json", "w"), indent=1, default=float)
    print(f"[{time.time()-t0:5.0f}s] listo")
