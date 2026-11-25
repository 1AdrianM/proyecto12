"""
Estudio de la "pregunta de decisión":
   ¿Qué pesos de penalización en la función de pérdida de la PINN garantizan que las predicciones
   cumplan la Segunda Ley (σ >= 0) frente a ruido aleatorio de instrumentación?

Se barren:   peso de la 2a Ley λ_S ∈ {0, 0.01, 0.1, 1, 10, 100}   +   variante "dura" (η = sigmoide).
             nivel de ruido ν ∈ {0.5, 1, 2, 5, 10} × ruido base,  con 3 % de picos de sensor (×0.6 a ×1.6).
Dos regímenes:
   S  (supervisado):    se entrena con η y ṁ "verdaderos" del simulador (fase de datos simulados).
   U  (sin etiquetas):  solo balances físicos + datos medidos (lo que habría con datos reales).
Resultado: results/estudio_segunda_ley.csv
"""
import sys, json, time
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, torch
from simulador import planta as P
from pinn import modelo as M

torch.set_num_threads(2)
CASO = "A"
NUS = [0.5, 1, 2, 5, 10]
LAMS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SEMILLAS = [0, 1]
EPOCAS = 100
filas = []


def banco_test(nu_lista, p_out, base_seed):
    return {nu: P.generar_dataset(CASO, 4000, seed=base_seed + i, nu=nu, p_outlier=p_out) for i, nu in enumerate(nu_lista)}


def registrar(regimen, config, seed, red):
    for nu, df in TEST.items():
        o, _ = M.evaluar(CASO, red, df)
        filas.append(dict(regimen=regimen, config=config, semilla=seed, nu=nu, **o))
    pd.DataFrame(filas).to_csv(RAIZ / "resultados" / "estudio_segunda_ley.csv", index=False)


TEST = banco_test(NUS, 0.03, 900)
t0 = time.time()

# ---------------- Régimen U: sin etiquetas (física + datos) ----------------
for seed in SEMILLAS:
    tr = P.generar_dataset(CASO, 20000, seed=100 + seed, nu=2.0, p_outlier=0.03)
    col = P.generar_dataset(CASO, 20000, seed=200 + seed, nu=6.0, p_outlier=0.15)
    for lam_s in LAMS:
        red = M.entrenar(CASO, tr, col, "soft", dict(D=0.0, E=1.0, M=1.0, S=lam_s), epocas=EPOCAS, seed=seed)
        registrar("U", f"soft lamS={lam_s}", seed, red)
        print(f"[{time.time()-t0:5.0f}s] U soft lamS={lam_s} seed={seed}", flush=True)
    red = M.entrenar(CASO, tr, col, "hard", dict(D=0.0, E=1.0, M=1.0, S=0.0), epocas=EPOCAS, seed=seed)
    registrar("U", "dura (sigmoide)", seed, red)
    print(f"[{time.time()-t0:5.0f}s] U hard seed={seed}", flush=True)

# ---------------- Régimen S: supervisado (datos simulados) ----------------
tr = P.generar_dataset(CASO, 20000, seed=300, nu=1.0)
col = P.generar_dataset(CASO, 20000, seed=301, nu=6.0, p_outlier=0.15)
configs = [("MLP sin física", "soft", dict(E=0.0, M=0.0, S=0.0)),
           ("PINN lamS=0", "soft", dict(S=0.0)),
           ("PINN lamS=10", "soft", dict(S=10.0)),
           ("PINN dura", "hard", dict(S=0.0))]
for nombre, modo, lam in configs:
    red = M.entrenar(CASO, tr, col, modo, lam, epocas=EPOCAS, seed=0)
    registrar("S", nombre, 0, red)
    print(f"[{time.time()-t0:5.0f}s] S {nombre}", flush=True)
print("listo")
