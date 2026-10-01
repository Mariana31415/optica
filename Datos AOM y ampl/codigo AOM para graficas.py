#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graficar.py - Genera las 3 figuras del informe a partir de datos.txt

Uso:
    python graficar.py              # lee datos.txt
    python graficar.py otro.txt     # lee otro archivo

Requiere:  pip install numpy scipy matplotlib
Salida:    carpeta ./figuras/  (PNG a 300 dpi y PDF vectorial)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import curve_fit

# ============================ OPCIONES ==============================
ARCHIVO        = sys.argv[1] if len(sys.argv) > 1 else "datos.txt"
CARPETA        = "figuras"
FORMATOS       = ("png", "pdf")
AJUSTAR_SIN2   = True    # ajustar el modelo sin^2 a la curva del AOM
EXTRAPOLAR     = True    # dibujar la prolongacion punteada mas alla de los datos
N_GANANCIA     = 4       # n de ultimos puntos (potencias altas) con ganancia estable
REF_LINEAL_DBM = -14     # Prf (dBm) donde se ancla la recta de pendiente 1 (log-log)
# ====================================================================

TEAL, CORAL, GRAY, DARK = "#0F6E56", "#C2502A", "#8A8A85", "#222222"
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "axes.titlesize": 12,
    "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 300,
})


# ---------------------------------------------------------------- lectura
def leer(ruta):
    """Devuelve (parametros, tablas). Secciones [nombre], lineas 'clave = valor'."""
    par, tablas, sec = {}, {}, None
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.split("#")[0].strip()
            if not linea:
                continue
            if linea.startswith("["):
                sec = linea.strip("[]").strip().lower()
                tablas[sec] = []
            elif "=" in linea:
                k, v = linea.split("=", 1)
                par[k.strip()] = float(v)
            else:
                tablas[sec].append([float(x) for x in linea.split()])
    return par, {k: np.array(v) for k, v in tablas.items() if v}


# ---------------------------------------------------------------- calculo
def recta(x, y):
    r = stats.linregress(x, y)
    return r.slope, r.intercept, r.stderr, r.intercept_stderr, r.rvalue ** 2


def sin2(rf, A, Pp, bg):
    """Modelo de Bragg: eta = bg + A * sin^2( (pi/2) * sqrt(P_RF / P_pi) )."""
    return bg + A * np.sin(np.pi / 2 * np.sqrt(rf / Pp)) ** 2


def ajustar_sin2(rf, eta):
    p0 = [min(1.0, eta.max() * 1.1), rf.max() * 1.2, eta.min()]
    cotas = ([0, rf.min(), 0], [1, np.inf, 0.05])
    po, pc = curve_fit(sin2, rf, eta, p0=p0, bounds=cotas, maxfev=50000)
    return po, np.sqrt(np.diag(pc))


def validar_extrapolacion(Prf, rf, eta):
    """Ajusta con los primeros n puntos y compara lo predicho con lo medido."""
    print("\nValidacion de la extrapolacion (ajusto con n puntos, predigo el resto):")
    print("  hasta Prf   eta_max   P_pi(mW)   prediccion / medido (%) de los puntos siguientes")
    for n in range(max(6, len(rf) - 5), len(rf)):
        try:
            po, _ = ajustar_sin2(rf[:n], eta[:n])
        except RuntimeError:
            continue
        pred = 100 * sin2(rf[n:], *po)
        txt = "  ".join(f"{p:.1f}/{100 * e:.1f}" for p, e in zip(pred, eta[n:]))
        print(f"  {Prf[n - 1]:+6.0f} dBm  {100 * (po[0] + po[2]):5.0f} %  {po[1]:9.0f}   {txt}")


def guardar(fig, nombre):
    os.makedirs(CARPETA, exist_ok=True)
    for ext in FORMATOS:
        fig.savefig(os.path.join(CARPETA, f"{nombre}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ================================================================ main
par, T = leer(ARCHIVO)
Pin, G, Zid = par["Pin_mW"], par["ganancia_dB"], par["Z_ideal_ohm"]
b_ideal = 10 * np.log10(Zid) - 30            # ordenada esperada: dBV = P + b_ideal

# ---- oscilador solo
P, V, sd = T["oscilador"][:, 0], T["oscilador"][:, 1], T["oscilador"][:, 2] / 1000
dBV = 20 * np.log10(V / 1000)
m, b, sm, sb, R2 = recta(P, dBV)
Z = 10 ** ((b + 30) / 10)
sZ = Z * np.log(10) / 10 * sb
Vid = lambda p: 1000 * np.sqrt(1e-3 * 10 ** (p / 10) * Zid)

# ---- oscilador + amplificador
Pa, Va = T["amplificador"][:, 0], T["amplificador"][:, 1]
dBVa = 20 * np.log10(Va / 1000)
ma, ba, *_ = recta(Pa, dBVa)
g_osc = dBVa - (m * Pa + b)                  # ganancia respecto al oscilador medido
g_ideal = dBVa - (Pa + b_ideal)              # ganancia respecto al caso ideal
G1, G2 = g_osc[-N_GANANCIA:].mean(), g_ideal[-N_GANANCIA:].mean()

# ---- AOM
Prf, Pm = T["aom"][:, 0], T["aom"][:, 1]
RF = 10 ** ((Prf + G) / 10)                  # potencia RF estimada en el AOM (mW)
eta = Pm / Pin                               # fraccion
fondo = Pm.min()

# ---------------------------------------------------------- resumen
print("=" * 62)
print(f"OSCILADOR   m = {m:.4f} +- {sm:.4f}   b = {b:.3f} +- {sb:.3f} dB   R2 = {R2:.5f}")
print(f"            Z = {Z:.1f} +- {sZ:.1f} ohm   (ideal {Zid:.0f} ohm, "
      f"desvio {b - b_ideal:+.2f} dB)")
print(f"AMPLIFICADOR  G = {G1:.1f} dB (resp. oscilador medido) | {G2:.1f} dB (resp. {Zid:.0f} ohm ideal)")
print(f"AOM         eta_max medida = {100 * eta.max():.1f} %  con RF estimada = {RF.max():.0f} mW "
      f"({10 * np.log10(RF.max()):.1f} dBm)")

fit_ok = False
if AJUSTAR_SIN2:
    try:
        (A, Pp, bg), (sA, sPp, sbg) = ajustar_sin2(RF, eta)
        fit_ok = True
        print(f"            sin2: A = {A:.3f} +- {sA:.3f}   P_pi = {Pp:.0f} +- {sPp:.0f} mW   "
              f"fondo = {100 * bg:.2f} %")
        print("            (las +- son solo estadisticas del ajuste; ver validacion abajo)")
        validar_extrapolacion(Prf, RF, eta)
    except RuntimeError:
        print("            No convergio el ajuste sin2.")
print("=" * 62)

# ================================================================ FIG 1
fig = plt.figure(figsize=(11.5, 5))
gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=.1, wspace=.28)
a = fig.add_subplot(gs[:, 0])
pp = np.linspace(P.min() - 2, P.max() + 2, 200)
a.plot(pp, Vid(pp), "--", color=GRAY, label=rf"Ideal, carga de {Zid:.0f} $\Omega$")
a.plot(pp, 1000 * 10 ** ((m * pp + b) / 20), color=TEAL, label="Ajuste a los datos")
a.errorbar(P, V, yerr=sd, fmt="o", color=TEAL, ms=5, capsize=2, label="Medidas")
a.set(xlabel="Potencia configurada $P$ (dBm)", ylabel=r"Voltaje medido $V_{RMS}$ (mV)",
      title="(a) Escala lineal: crecimiento exponencial")
a.legend(loc="upper left")
a.text(.97, .06, "+6 dB  →  voltaje ×2\n+20 dB →  voltaje ×10",
       transform=a.transAxes, ha="right", fontsize=10, color=DARK)

b_ = fig.add_subplot(gs[0, 1])
r = fig.add_subplot(gs[1, 1], sharex=b_)
b_.plot(pp, pp + b_ideal, "--", color=GRAY,
        label=rf"Ideal {Zid:.0f} $\Omega$: pendiente 1, $b={b_ideal:.2f}$")
b_.plot(pp, m * pp + b, color=TEAL, label="Ajuste lineal")
b_.errorbar(P, dBV, yerr=20 / np.log(10) * sd / V, fmt="o", color=TEAL, ms=5, capsize=2)
b_.set(ylabel=r"$V_{RMS}$ (dBV)", title="(b) Escala en dB: relación lineal")
b_.legend(loc="upper left", fontsize=9.5)
b_.text(.97, .1, f"$m={m:.3f}\\pm{sm:.3f}$\n$b={b:.3f}\\pm{sb:.3f}$ dB\n$R^2={R2:.4f}$\n"
        f"$Z={Z:.1f}\\pm{sZ:.1f}\\ \\Omega$", transform=b_.transAxes, ha="right", fontsize=10)
plt.setp(b_.get_xticklabels(), visible=False)
r.axhline(0, color=GRAY, lw=1)
r.plot(P, dBV - (m * P + b), "o", color=TEAL, ms=4)
r.set(xlabel="Potencia configurada $P$ (dBm)", ylabel="Residuo (dB)", ylim=(-.3, .3))
guardar(fig, "fig1_oscilador")

# ================================================================ FIG 2
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"wspace": .28})
x = np.linspace(Pa.min() - 2, P.max() + 2, 100)
ax[0].plot(x, m * x + b, "--", color=TEAL, lw=1.2, label="Oscilador solo (ajuste, extrapolado)")
ax[0].plot(P, dBV, "o", color=TEAL, ms=4, label="Oscilador solo (medidas)")
xa = x[x <= Pa.max() + 2]
ax[0].plot(xa, ma * xa + ba, color=CORAL, lw=1.5)
ax[0].plot(Pa, dBVa, "s", color=CORAL, ms=5, label="Oscilador + amplificador")
x0 = Pa[-1]
ax[0].annotate("", xy=(x0, dBVa[-1]), xytext=(x0, m * x0 + b),
               arrowprops=dict(arrowstyle="<->", color=DARK))
ax[0].text(x0 + 1.5, (dBVa[-1] + m * x0 + b) / 2, f"$G\\approx{G1:.0f}$ dB", fontsize=11, color=DARK)
ax[0].set(xlabel="Potencia configurada $P$ (dBm)", ylabel=r"$V_{RMS}$ (dBV)",
          title="(a) Voltaje con y sin amplificador")
ax[0].legend(loc="lower right", fontsize=9.5)

ax[1].plot(Pa, g_osc, "s-", color=CORAL, ms=5, lw=.8, label="Respecto al oscilador medido")
ax[1].plot(Pa, g_ideal, "o-", color=GRAY, ms=5, lw=.8, label=rf"Respecto a {Zid:.0f} $\Omega$ ideal")
ax[1].axhline(G1, color=CORAL, ls=":", lw=1)
ax[1].axhline(G2, color=GRAY, ls=":", lw=1)
ax[1].text(Pa[-N_GANANCIA] + .2, G1 + .14, f"{G1:.1f} dB", color=CORAL, fontsize=10)
ax[1].text(Pa[-N_GANANCIA] + .2, G2 - .3, f"{G2:.1f} dB", color="#555", fontsize=10)
ax[1].set(xlabel="Potencia configurada $P$ (dBm)", ylabel="Ganancia (dB)",
          title="(b) Ganancia del amplificador",
          ylim=(g_ideal.min() - .6, g_osc.max() + .5))
ax[1].legend(loc="upper right", fontsize=9.5)
guardar(fig, "fig2_amplificador")

# ================================================================ FIG 3
fig = plt.figure(figsize=(11.5, 5))
gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=.1, wspace=.28)
a = fig.add_subplot(gs[0, 0] if fit_ok else gs[:, 0])
xmax = RF.max() * 1.7
if fit_ok:
    xx = np.linspace(1, xmax, 600)
    dentro = xx <= RF.max()
    a.plot(xx[dentro], 100 * sin2(xx[dentro], A, Pp, bg), color=CORAL,
           label="Ajuste $\\mathrm{sin}^2$ (modelo Bragg)")
    if EXTRAPOLAR:
        fuera = xx >= RF.max()
        a.plot(xx[fuera], 100 * sin2(xx[fuera], A, Pp, bg), "--", color=CORAL, alpha=.6,
               label="Extrapolación (no medida)")
        if Pp < xmax:
            a.axvline(Pp, color=GRAY, ls=":", lw=1)
            a.text(Pp + .03 * xmax, .04 * 100 * eta.max(),
                   f"máximo estimado\n(no verificado)\n≈{Pp / 1000:.1f} W, "
                   f"$\\eta\\approx{100 * (A + bg):.0f}$ %", fontsize=9.5, color="#555")
a.plot(RF, 100 * eta, "o", color=DARK, ms=5, label="Medidas")
a.set(ylabel=r"Eficiencia $\eta=P/P_{in}$ (%)", title="(a) Eficiencia vs potencia RF",
      xlim=(0, xmax if EXTRAPOLAR and fit_ok else RF.max() * 1.1), ylim=(0, 100 * eta.max() * 1.15))
a.legend(loc="center right" if fit_ok and EXTRAPOLAR else "upper left", fontsize=9.5)
if fit_ok:
    plt.setp(a.get_xticklabels(), visible=False)
    r = fig.add_subplot(gs[1, 0], sharex=a)
    r.axhline(0, color=GRAY, lw=1)
    r.plot(RF, 100 * (eta - sin2(RF, A, Pp, bg)), "o", color=DARK, ms=4)
    r.set(xlabel="Potencia RF estimada en el AOM (mW)", ylabel="Residuo (pp)")
else:
    a.set_xlabel("Potencia RF estimada en el AOM (mW)")

b_ = fig.add_subplot(gs[:, 1])
b_.loglog(RF, 100 * eta, "o", color=DARK, ms=5)
i = int(np.argmin(abs(Prf - REF_LINEAL_DBM)))
k = eta[i] / RF[i]
xl = np.array([RF[i] / 9, RF[i] * 9])
b_.plot(xl, 100 * k * xl, "--", color=CORAL, label="Pendiente 1: régimen lineal")
b_.axhline(100 * fondo / Pin, color=GRAY, ls=":", label=f"Fondo ≈ {fondo:.2f} mW")
b_.set(xlabel="Potencia RF estimada en el AOM (mW)", ylabel=r"Eficiencia $\eta$ (%)",
       title="(b) Escala log-log: tres regímenes")
b_.legend(loc="upper left", fontsize=9.5)
for txt, (fx, fy) in {"fondo": (.03, .20), "lineal": (.47, .21), "saturación": (.80, .64)}.items():
    b_.text(fx, fy, txt, transform=b_.transAxes, fontsize=10, color="#555")
guardar(fig, "fig3_aom")

print(f"Figuras guardadas en ./{CARPETA}/  ({', '.join(FORMATOS)})")