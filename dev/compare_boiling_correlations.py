# -*- coding: utf-8 -*-
"""
Compare the nucleate pool boiling correlations in ht.boiling_nucleic on a
copper tube with Ra = 1 um, for

* water at a saturation temperature of 100 C, and
* R-134a at a saturation temperature of 5 C.

Two sweeps are made for each fluid:

1. Heat flux q from 20 to 100 kW/m^2 (x axis q in kW/m^2).
2. Excess temperature Te (x axis Te in K). The Te range for each fluid is the
   range of Te = q/h that the correlations predict over sweep 1, so both sweeps
   cover the same operating region.

3. Boiling curve: heat flux q = h*Te (y axis, kW/m^2) against Te from sweep 2.

Sweeps 2 and 3 stop each curve where q exceeds the critical heat flux (Zuber).
For each sweep and fluid two plots are saved in this folder, with linear and
log-log axes. The heat transfer coefficient is in kW/m^2/K. Fluid properties
come from CoolProp.
"""

import os
import warnings

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import ht

# Data-quality and Te-sensitivity warnings from Gorenflo are not needed here
warnings.simplefilter("ignore")


#%% Inputs
q_kW = np.linspace(20.0, 100.0, 81)   # kW/m^2
q_range = q_kW*1E3                    # W/m^2
n_Te = 81                             # number of points in the Te sweep

Ra = 1.0E-6          # m, arithmetic-mean roughness (Ra = 1 um)
# Cooper uses the older DIN 4762 roughness Rp; Ra = 0.4*Rp (Gorenflo et al. 2014, Eq. 8)
Rp = Ra/0.4          # m

cases = [
    # (CoolProp name, label for titles/files, Tsat [K], CASRN,
    #  Stephan-Abdelsalam variant, Rohsenow Csf, Rohsenow n)
    ("Water",  "Water",  100.0 + 273.15, "7732-18-5", "water",       0.013, 1.0),
    ("R134a",  "R-134a",   5.0 + 273.15, "811-97-2",  "refrigerant", 0.013, 1.7),
]

out_dir = os.path.dirname(os.path.abspath(__file__))


#%% Fluid properties at saturation
def saturation_properties(fluid, Tsat):
    P = CP.PropsSI("P", "T", Tsat, "Q", 0, fluid)
    return dict(
        P=P,
        Pc=CP.PropsSI("Pcrit", fluid),
        MW=CP.PropsSI("M", fluid)*1E3,                       # g/mol
        rhol=CP.PropsSI("D", "T", Tsat, "Q", 0, fluid),
        rhog=CP.PropsSI("D", "T", Tsat, "Q", 1, fluid),
        mul=CP.PropsSI("V", "T", Tsat, "Q", 0, fluid),
        kl=CP.PropsSI("L", "T", Tsat, "Q", 0, fluid),
        Cpl=CP.PropsSI("C", "T", Tsat, "Q", 0, fluid),
        Hvap=(CP.PropsSI("H", "T", Tsat, "Q", 1, fluid)
              - CP.PropsSI("H", "T", Tsat, "Q", 0, fluid)),
        sigma=CP.PropsSI("I", "T", Tsat, "Q", 0, fluid),
    )


#%% Correlations
def forster_zuber(fluid, Tsat, p, q=None, Te=None):
    # Forster-Zuber needs dPsat = Psat(Tsat + Te) - Psat(Tsat). With Te given
    # this is direct; with q given, Te is unknown, so solve q = h(Te)*Te.
    def h_of_Te(Te):
        dPsat = CP.PropsSI("P", "T", Tsat + Te, "Q", 0, fluid) - p["P"]
        return ht.Forster_Zuber(Te=Te, dPsat=dPsat, rhol=p["rhol"], rhog=p["rhog"],
                                mul=p["mul"], kl=p["kl"], Cpl=p["Cpl"],
                                Hvap=p["Hvap"], sigma=p["sigma"])
    if Te is None:
        Te = brentq(lambda Te: h_of_Te(Te)*Te - q, 1E-3, 80.0)
    return h_of_Te(Te)


def all_correlations(fluid, Tsat, CASRN, sa_variant, Csf, n_rohsenow, p,
                     q=None, Te=None):
    # Exactly one of q or Te is given; every correlation is called with it.
    liquid = dict(rhol=p["rhol"], rhog=p["rhog"], kl=p["kl"], Cpl=p["Cpl"],
                  Hvap=p["Hvap"], sigma=p["sigma"])
    drive = dict(q=q, Te=Te)
    return {
        "Gorenflo (2010)": ht.Gorenflo(p["P"], p["Pc"], CASRN=CASRN, Ra=Ra, **drive),
        "Cooper": ht.Cooper(P=p["P"], Pc=p["Pc"], MW=p["MW"], Rp=Rp, **drive),
        "Stephan-Abdelsalam": ht.Stephan_Abdelsalam(
            mul=p["mul"], Tsat=Tsat, correlation=sa_variant, **liquid, **drive),
        "Rohsenow": ht.Rohsenow(mul=p["mul"], Csf=Csf, n=n_rohsenow, **liquid, **drive),
        "Forster-Zuber": forster_zuber(fluid, Tsat, p, **drive),
        "McNelly": ht.McNelly(P=p["P"], **liquid, **drive),
        "HEDH-Taborek": ht.HEDH_Taborek(P=p["P"], Pc=p["Pc"], **drive),
        "Montinsky": ht.Montinsky(P=p["P"], Pc=p["Pc"], **drive),
        "Bier": ht.Bier(P=p["P"], Pc=p["Pc"], **drive),
    }


def sweep(case, p, q_values=None, Te_values=None):
    fluid, label, Tsat, CASRN, sa_variant, Csf, n_rohsenow = case
    results = {name: [] for name in style}
    if q_values is not None:
        points = [dict(q=q) for q in q_values]
    else:
        points = [dict(Te=Te) for Te in Te_values]
    for point in points:
        for name, h in all_correlations(fluid, Tsat, CASRN, sa_variant, Csf,
                                        n_rohsenow, p, **point).items():
            results[name].append(h)
    return {name: np.array(h) for name, h in results.items()}


#%% Plot style
# Categorical slots in fixed order. Montinsky (Mostinski) and HEDH-Taborek are
# the same family of correlation, so they share a hue and differ by line style.
style = {
    "Gorenflo (2010)":    dict(color="#2a78d6", linestyle="-"),
    "Cooper":             dict(color="#eb6834", linestyle="-"),
    "Stephan-Abdelsalam": dict(color="#1baf7a", linestyle="-"),
    "Rohsenow":           dict(color="#eda100", linestyle="-"),
    "Forster-Zuber":      dict(color="#e87ba4", linestyle="-"),
    "McNelly":            dict(color="#008300", linestyle="-"),
    "HEDH-Taborek":       dict(color="#4a3aa7", linestyle="-"),
    "Montinsky":          dict(color="#4a3aa7", linestyle="--"),
    "Bier":               dict(color="#e34948", linestyle="-"),
}
ink_primary, ink_secondary, ink_muted = "#0b0b0b", "#52514e", "#898781"
grid_color, axis_color, surface = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def log_ticks(lo, hi, axis):
    if axis == "x":
        # x spans less than a decade, so label it densely, within the data
        candidates = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100]
        return [t for t in candidates if lo - 1E-9 <= t <= hi + 1E-9]
    # y can span two decades: 1-2-5 steps keep the labels apart
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    return [t for t in candidates if 0.8*lo <= t <= 1.25*hi]


def make_plot(x, results, xlabel, title, log, path,
              ylabel="Heat transfer coefficient, h [kW/m²·K]", q_max=None):
    # results are in SI units (W/m^2/K or W/m^2) and are plotted in kilo-units.
    # q_max, if given, is drawn as a horizontal reference line [W/m^2].
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    # Legend lists the lines from top to bottom at the right-most point where
    # every curve still exists (curves can stop at the critical heat flux);
    # each correlation keeps its own colour whatever its position.
    all_finite = np.all([np.isfinite(h) for h in results.values()], axis=0)
    i_rank = np.nonzero(all_finite)[0][-1]
    order = sorted(results, key=lambda name: -results[name][i_rank])
    for name in order:
        ax.plot(x, results[name]/1E3, linewidth=2.0, label=name, **style[name])
    if q_max is not None:
        ax.axhline(q_max/1E3, color=ink_muted, linewidth=1.0, linestyle=":")
        ax.text(x[0], q_max/1E3, f" Critical heat flux (Zuber) = {q_max/1E3:.0f} kW/m²",
                color=ink_secondary, fontsize=8, va="bottom", ha="left")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        h_min = min(np.nanmin(h) for h in results.values())/1E3
        h_max = max(np.nanmax(h) for h in results.values())/1E3
        if q_max is not None:
            h_max = max(h_max, q_max/1E3)
        ax.set_xticks(log_ticks(x[0], x[-1], "x"))
        ax.set_yticks(log_ticks(h_min, h_max, "y"))
        for axis in (ax.get_xaxis(), ax.get_yaxis()):
            axis.set_major_formatter(plt.FormatStrFormatter("%g"))
            axis.set_minor_formatter(plt.NullFormatter())
    else:
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel, color=ink_secondary)
    ax.set_ylabel(ylabel, color=ink_secondary)
    ax.set_title(title, color=ink_primary, loc="left", fontsize=12)
    ax.grid(True, which="major", color=grid_color, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_color)
    ax.tick_params(colors=ink_muted, which="both")
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                       frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(ink_secondary)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    print("saved", path)


def print_table(results, x_name, x_values, x_unit):
    print(f"  {'correlation':<20}{'h at ' + x_name + ' = ' + format(x_values[0], 'g') + ' ' + x_unit:>22}"
          f"{'h at ' + x_name + ' = ' + format(x_values[-1], 'g') + ' ' + x_unit:>24}  [kW/m2/K]")
    for name, h in results.items():
        print(f"  {name:<20}{h[0]/1E3:>22.2f}{h[-1]/1E3:>24.2f}")


#%% Run
for case in cases:
    fluid, label, Tsat = case[0], case[1], case[2]
    p = saturation_properties(fluid, Tsat)
    title = (f"{label} pool boiling, Tsat = {Tsat - 273.15:.0f} °C, "
             f"copper tube, Ra = 1 µm")
    stem = os.path.join(out_dir, "boiling_" + label.replace("-", "").lower())
    print(f"\n{label}: Tsat = {Tsat - 273.15:.0f} C, P = {p['P']/1E3:.1f} kPa, "
          f"p* = {p['P']/p['Pc']:.4f}")

    # Sweep 1: heat flux on the x axis
    results_q = sweep(case, p, q_values=q_range)
    print_table(results_q, "q", q_kW, "kW/m2")
    xlabel = "Heat flux, q [kW/m²]"
    make_plot(q_kW, results_q, xlabel, title, log=False, path=stem + "_linear.png")
    make_plot(q_kW, results_q, xlabel, title + " (log-log)", log=True,
              path=stem + "_loglog.png")

    # Sweep 2: excess temperature on the x axis, over the Te range the
    # correlations predict for 20-100 kW/m^2
    Te_predicted = np.concatenate([q_range/h for h in results_q.values()])
    Te_values = np.linspace(np.floor(Te_predicted.min()),
                            np.ceil(Te_predicted.max()), n_Te)
    results_Te = sweep(case, p, Te_values=Te_values)
    # Nucleate boiling ends at the critical heat flux; blank each curve where
    # q = h*Te exceeds it (Zuber, K = 0.149 as in Lienhard & Dhir / Incropera).
    q_max = ht.Zuber(sigma=p["sigma"], Hvap=p["Hvap"], rhol=p["rhol"],
                     rhog=p["rhog"], K=0.149)
    print(f"  Critical heat flux (Zuber) = {q_max/1E3:.0f} kW/m2; "
          "points above it are not plotted")
    print_table(results_Te, "Te", Te_values, "K")
    for name, h in results_Te.items():
        h[h*Te_values > q_max] = np.nan
    xlabel = "Excess temperature, ΔTe = Tw − Tsat [K]"
    make_plot(Te_values, results_Te, xlabel, title, log=False,
              path=stem + "_Te_linear.png")
    make_plot(Te_values, results_Te, xlabel, title + " (log-log)", log=True,
              path=stem + "_Te_loglog.png")

    # Boiling curve: heat flux q = h*Te against excess temperature (same sweep)
    results_curve = {name: h*Te_values for name, h in results_Te.items()}
    ylabel = "Heat flux, q [kW/m²]"
    make_plot(Te_values, results_curve, xlabel, title + ", boiling curve",
              log=False, path=stem + "_curve_linear.png", ylabel=ylabel,
              q_max=q_max)
    make_plot(Te_values, results_curve, xlabel, title + ", boiling curve (log-log)",
              log=True, path=stem + "_curve_loglog.png", ylabel=ylabel,
              q_max=q_max)
