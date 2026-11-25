"""
PINN (Red Neuronal Informada por la Física) del gemelo digital.

Idea en una frase: una red neuronal estima la eficiencia real de la turbina (y por tanto el nivel de
ensuciamiento por sílice) a partir de los sensores, pero su función de pérdida le exige cumplir:

  1) Balance de energía (1a Ley):    W_neta_física(η̂, ṁ) = W_neta_medida
  2) Balance de masa:                ṁ̂_vapor = ṁ_vapor calculado por los flash (balance m, h)
  3) 2a Ley (generación de entropía): σ_k = s_salida - s_entrada >= 0 en cada tramo de la turbina
        con s_salida calculada de las tablas IAPWS-95 (relación Tds / Clapeyron: h_fg = T s_fg)
  4) Supervisión con datos simulados (η y ṁ "verdaderos")

Las ecuaciones físicas se reutilizan de simulador/planta.py con el backend de PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn

from simulador import planta as P

TB = P.hacer_torch_backend(torch.float32)
AG = P.Agua(TB)
COLS = P.COLS_MEDIDAS            # m_brine, T_brine, y_ncg, Tcw, P1, P2, Pc, W_neta, m_vapor


# ----------------------------------------------------------------------------------------------
# Capa física diferenciable
# ----------------------------------------------------------------------------------------------
def capa_fisica(caso, med, eta, m_hat, par=None):
    """
    med: dict de tensores con las mediciones (columnas COLS); eta: η̂ estimado; m_hat: ṁ_vapor estimado.
    Devuelve potencia neta física, flujo de vapor por balance, σ de cada tramo y eficiencia 'limpia'.
    """
    par = par or P.PAR
    w = par["w_nacl_B"] if caso == "B" else 0.0
    if caso == "A":
        h_in = AG.pT("hf", med["T_brine"])
    else:
        h_in = (1.0 - w) * AG.pT("hf", med["T_brine"]) + w * par["cs_nacl"] * med["T_brine"]
    m_hp, m_l1, h_l1, w_l1, _ = P.flash(TB, AG, med["P1"], med["m_brine"], h_in, w, par, caso)
    m_lp, m_l2, _, _, _ = P.flash(TB, AG, med["P2"], m_l1, h_l1, w_l1 if caso == "B" else w, par, caso)
    m_bal = m_hp + m_lp                                          # balance de masa-energía en los flash
    r1 = P.etapa_ref(TB, AG, med["P1"], med["Pc"])
    r2 = P.etapa_ref(TB, AG, med["P2"], med["Pc"])
    esc = m_hat / m_bal
    dh_pond = (m_hp * r1["dh_is"] + m_lp * r2["dh_is"]) * esc
    W_bruta = par["eta_gen"] * eta * dh_pond
    T_c = AG.Tsat(torch.clamp(med["Pc"] - par["k_ej"] * med["y_ncg"], min=0.02))
    W_aux = P.auxiliares(TB, AG, W_bruta, m_hat, med["y_ncg"], med["Pc"], T_c, par, caso)
    h1o, _, _, sig1 = P.estado_salida(r1, eta)
    h2o, _, _, sig2 = P.estado_salida(r2, eta)
    eta_cl = (m_hp * r1["dh_is"] * P.eta_limpia(par, r1["x_is"]) + m_lp * r2["dh_is"] * P.eta_limpia(par, r2["x_is"])) \
        / (m_hp * r1["dh_is"] + m_lp * r2["dh_is"])
    return dict(W_neta=W_bruta - W_aux, m_bal=m_bal, m_hp=m_hp, m_lp=m_lp, h_exh=(m_hp * h1o + m_lp * h2o) / m_bal, sig1=sig1, sig2=sig2, eta_limpia=eta_cl,
                T_c=T_c, dTds_resid=(r1["hfg"] - (AG.Tsat(med["Pc"]) + 273.15) * r1["sfg"]))   # identidad Tds: h_fg = T s_fg


# ----------------------------------------------------------------------------------------------
# Red
# ----------------------------------------------------------------------------------------------
class RedPINN(nn.Module):
    def __init__(self, n_in=len(COLS), n_h=64, modo="soft"):
        super().__init__()
        self.modo = modo
        self.net = nn.Sequential(nn.Linear(n_in, n_h), nn.SiLU(), nn.Linear(n_h, n_h), nn.SiLU(),
                                 nn.Linear(n_h, n_h), nn.SiLU(), nn.Linear(n_h, 2))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))

    def forward(self, x_raw, m_bal):
        z = self.net((x_raw - self.mu) / self.sd)
        if self.modo == "hard":           # η ∈ (0,1) por construcción (restricción "dura")
            eta = torch.sigmoid(z[:, 0] + 1.1)
        else:                             # sin restricción estructural
            eta = 0.75 + 0.1 * z[:, 0]
        m_hat = m_bal * (1.0 + 0.05 * torch.tanh(z[:, 1]))
        return eta, m_hat


def a_tensores(df, cols_sufijo="_med"):
    return {c: torch.as_tensor(np.array(df[c + cols_sufijo].values), dtype=torch.float32) for c in COLS}


def matriz(med):
    return torch.stack([med[c] for c in COLS], dim=1)


def balance_masa(caso, med, par=None):
    par = par or P.PAR
    r = capa_fisica(caso, med, torch.full_like(med["m_brine"], 0.75), med["m_vapor"], par)
    return r["m_bal"]


# ----------------------------------------------------------------------------------------------
# Pérdidas y entrenamiento
# ----------------------------------------------------------------------------------------------
ESC = dict(W=0.01 * 25000.0, m=0.015 * 70.0, sig=0.05, eta=0.03, mm=2.0)


def perdidas(caso, red, med, y_true=None, lam=None, par=None):
    lam = lam or {}
    x = matriz(med)
    m_bal = balance_masa(caso, med, par)
    eta, m_hat = red(x, m_bal)
    fis = capa_fisica(caso, med, eta, m_hat, par)
    L = {}
    if y_true is not None:
        L["D"] = (((eta - y_true["eta"]) / ESC["eta"]) ** 2).mean() + (((m_hat - y_true["m"]) / ESC["mm"]) ** 2).mean()
    L["E"] = (((fis["W_neta"] - med["W_neta"]) / ESC["W"]) ** 2).mean()
    L["M"] = (((m_hat - fis["m_bal"]) / ESC["m"]) ** 2).mean()
    L["S"] = (((torch.relu(-fis["sig1"]) + torch.relu(-fis["sig2"])) / ESC["sig"]) ** 2).mean() \
        + (torch.relu(-eta) ** 2).mean() / ESC["eta"] ** 2
    total = sum(lam.get(k, 0.0) * v for k, v in L.items())
    return total, L, eta, m_hat, fis


def entrenar(caso, df_train, df_col, modo="soft", lam=None, epocas=250, bs=1024, lr=3e-3, seed=0, verbose=False, par=None):
    """
    df_train: datos etiquetados (con _med y _true).  df_col: puntos de colocación sin etiquetas
    (entradas extremas / con picos de sensor) donde solo se evalúan las pérdidas físicas.
    lam: pesos {'D','E','M','S'}.
    """
    lam = dict(dict(D=1.0, E=1.0, M=1.0, S=0.0), **(lam or {}))
    torch.manual_seed(seed)
    np.random.seed(seed)
    red = RedPINN(modo=modo)
    med_tr = a_tensores(df_train)
    X = matriz(med_tr)
    red.mu.copy_(X.mean(0))
    red.sd.copy_(X.std(0) + 1e-9)
    y_tr = dict(eta=torch.as_tensor(np.array(df_train["eta_ef_true"].values), dtype=torch.float32),
                m=torch.as_tensor(np.array(df_train["m_vapor_true"].values), dtype=torch.float32))
    med_co = a_tensores(df_col) if df_col is not None else None
    n = len(df_train)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    hist = []
    for ep in range(epocas):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            mb = {k: v[idx] for k, v in med_tr.items()}
            yb = {k: v[idx] for k, v in y_tr.items()}
            tot, L, *_ = perdidas(caso, red, mb, yb, lam, par)
            if med_co is not None and lam["S"] > 0:                    # física en puntos de colocación
                idc = torch.randint(0, len(df_col), (bs,))
                mc = {k: v[idc] for k, v in med_co.items()}
                _, Lc, *_ = perdidas(caso, red, mc, None, lam, par)
                tot = tot + lam["S"] * Lc["S"] + 0.0 * lam["E"] * Lc["E"]
            opt.zero_grad()
            tot.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 5.0)
            opt.step()
        sch.step()
        if verbose and ep % 25 == 0:
            hist.append((ep, float(tot)))
            print(f"  época {ep:4d}  pérdida {float(tot):.4f}  " + "  ".join(f"{k}={float(v):.3f}" for k, v in L.items()))
    return red


@torch.no_grad()
def evaluar(caso, red, df, par=None):
    """Métricas sobre un DataFrame con _med y _true."""
    red.eval()
    med = a_tensores(df)
    tot, L, eta, m_hat, fis = perdidas(caso, red, med, None, {}, par)
    b = (P.PAR if par is None else par)["b_foul"]
    phi_hat = (1.0 - eta / fis["eta_limpia"]) / b
    eta_t = torch.as_tensor(np.array(df["eta_ef_true"].values), dtype=torch.float32)
    phi_t = torch.as_tensor(np.array(df["phi_true"].values), dtype=torch.float32)
    viol = ((fis["sig1"] < 0) | (fis["sig2"] < 0) | (eta < 0)).float()
    out = dict(
        rmse_eta=float(torch.sqrt(((eta - eta_t) ** 2).mean())),
        mae_phi=float((phi_hat - phi_t).abs().mean()),
        err_rel_W_pct=float((100 * (fis["W_neta"] - torch.as_tensor(np.array(df["W_neta_true"].values), dtype=torch.float32)).abs()
                             / torch.as_tensor(np.array(df["W_neta_true"].values), dtype=torch.float32)).mean()),
        err_rel_m_pct=float((100 * (m_hat - torch.as_tensor(np.array(df["m_vapor_true"].values), dtype=torch.float32)).abs()
                             / torch.as_tensor(np.array(df["m_vapor_true"].values), dtype=torch.float32)).mean()),
        err_rel_h_pct=float((100 * (fis["h_exh"] - torch.as_tensor(np.array(df["h_exh_true"].values), dtype=torch.float32)).abs()
                             / torch.as_tensor(np.array(df["h_exh_true"].values), dtype=torch.float32)).mean()),
        viol_pct=float(100 * viol.mean()),
        sigma_min=float(torch.minimum(fis["sig1"], fis["sig2"]).min()),
    )
    red.train()
    return out, dict(eta=eta.numpy(), phi_hat=phi_hat.numpy(), m_hat=m_hat.numpy(), W_fis=fis["W_neta"].numpy(),
                     sig1=fis["sig1"].numpy(), sig2=fis["sig2"].numpy(), h_exh=fis["h_exh"].numpy(), eta_limpia=fis["eta_limpia"].numpy())


def guardar(red, ruta, caso):
    torch.save(dict(state=red.state_dict(), modo=red.modo, caso=caso), ruta)


def cargar(ruta):
    d = torch.load(ruta, map_location="cpu")
    red = RedPINN(modo=d["modo"])
    red.load_state_dict(d["state"])
    red.eval()
    return red, d["caso"]
