"""
Separador ciclónico (modelo de Lapple, 1951) para la separación bifásica centrífuga tras cada flash.

    d50 = sqrt( 9 μ_g W / (2 π N_e v_i (ρ_l - ρ_g)) )        diámetro de corte (50 % de captura)
    η(d) = 1 / (1 + (d50/d)²)                                  eficiencia fraccional

Geometría estándar de Lapple: ancho de entrada W = D/4, altura H = D/2  ->  A_in = D²/8.
Sirve para dimensionar el separador y estimar qué tamaño de gotas de salmuera (y sílice disuelta)
puede arrastrar el vapor hacia la turbina.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI


def lapple(P_bar, m_vapor, v_in=20.0, N_e=5.0):
    P = P_bar * 1e5
    rho_g = PropsSI("D", "P", P, "Q", 1, "Water")
    rho_l = PropsSI("D", "P", P, "Q", 0, "Water")
    mu_g = PropsSI("V", "P", P, "Q", 1, "Water")
    Q = m_vapor / rho_g                       # m3/s de vapor
    A_in = Q / v_in
    D = np.sqrt(8.0 * A_in)                   # A_in = D^2 / 8
    W = D / 4.0
    d50 = np.sqrt(9.0 * mu_g * W / (2.0 * np.pi * N_e * v_in * (rho_l - rho_g)))
    return dict(P_bar=P_bar, m_vapor=m_vapor, Q_m3s=Q, D_m=D, W_m=W, d50_um=d50 * 1e6, rho_g=rho_g, rho_l=rho_l, mu_g=mu_g)


def eficiencia_fraccional(d_um, d50_um):
    d = np.asarray(d_um, float)
    return 1.0 / (1.0 + (d50_um / d) ** 2)


if __name__ == "__main__":
    for P_bar, m in ((6.5, 47.2), (1.8, 23.9)):
        print({k: round(v, 4) for k, v in lapple(P_bar, m).items()})
