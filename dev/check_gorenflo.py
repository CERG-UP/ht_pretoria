# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:29:23 2026

@author: Work at Home
"""

import warnings


#%% Inputs
fluid = None #"Water"

P_abs = 101.325       # kPa
P_c = 22090.0         # kPa

# Given Parameters
T_s = 115.0 + 273.15          # C (Initial bearing temperature)
T_sat = 100.0  + 273.15       # C (Saturated water pool at 1 atm)
R_a = 0.1e-6          # m (Surface roughness, 0.1 um)

# Excess Temperature
dT_e = T_s - T_sat


# Saturation-curve slope and surface tension at the reference reduced
# pressure p* = 0.1 (not at the operating pressure). Only used to estimate h0
# for a fluid not in the tables. Water at p* = 0.1 (217.4 C), VDI Heat Atlas
# Table H2.1. (The Cengel and Ghajar value of 38210 Pa/K is at 100 C.)
dp_dT_ref = 38210 # 42694.0   # Pa/K
dp_dT = dp_dT_ref

sigma_ref = 0.03367   # N/m

e_cu = 35350.0        # W s^0.5/m^2/K (copper effusivity; stainless steel ~8450)
e_ss = 8450
#%% Conversion
P = P_abs
Pc = P_c

dPdT = dp_dT

sigma = sigma_ref
q= None
Te = dT_e
h0 = None
CASRN=None
# fluid is set in Inputs
Ra = R_a
eff = e_ss
#              return_CASRN: bool=False

#%% Reference values for horizontal copper tube
Ra0 = 0.4E-6    #m - Roughness
q0 = 2E4        # W/m2 - heat flux
eff_Cu = 35350.0  # W s^0.5 m^-2 K^-1, thermal effusivity of copper

#Intermediate calcs
Pr = P/Pc

#%% Database of measured reference HTCs

h0_given = h0 is not None  #Was a reference heat trasnfer coefficient given? if not, we'll use one from the tables


# VDI Heat Atlas 2nd edition (Gorenflo 2010) — reference heat transfer coefficients.
# Keys are common English names (from gorenflo_casrn_to_name). For internal CASRN-keyed
# lookup used inside Gorenflo(), see _h0_VDI_2e_by_casrn (built after gorenflo_casrn_to_name).
h0_VDI_2e = {
# Hydrocarbons — light
"Methane": 7200.0, "Ethylene": 4200.0, "Ethane": 4600.0,
"Propylene": 4200.0, "Propane": 4300.0,
# Hydrocarbons — n-alkanes / branched
"n-Butane": 3600.0, "IsoButane": 3700.0,
"n-Pentane": 3300.0, "Isopentane": 3200.0,
"n-Hexane": 3200.0, "CycloHexane": 3000.0,
"n-Heptane": 2900.0,
# Aromatics
"Benzene": 2900.0, "Toluene": 2800.0, "Biphenyl": 2100.0,
# Alcohols
"Methanol": 5400.0, "Ethanol": 4350.0, "n-Propanol": 3750.0,
"Isopropanol": 4100.0, "n-Butanol": 2600.0, "IsoButanol": 4500.0,
"2-Butanol": 3400.0,
# Other organics
"Acetaldehyde": 3500.0, "Acetone": 3300.0,
# Refrigerants — HFCs / HCFCs / CFCs / natural
"CO2": 5500.0, "R23": 4800.0, "R32": 5000.0, "R125": 4400.0,
"R134a": 4200.0, "R143a": 4700.0, "R152a": 4600.0, "R1234yf": 3000.0,
"R227EA": 4100.0, "RC318": 4200.0, "R14": 4750.0, "R123": 3000.0,
"R11": 2800.0, "R12": 4000.0, "R13": 3900.0, "R13B1": 3500.0,
"R22": 3900.0, "R113": 2650.0, "R114": 3800.0, "R115": 4200.0,
"R40": 4400.0, "R10": 3200.0, "SF6": 3700.0, "R502": 3300.0,
# Common process fluids
"Water": 5600.0, "Ammonia": 7000.0,
# Cryogenics (helium value is at q0 = 1 kW/m², footnote i)
"Oxygen": 9500.0, "Nitrogen": 10000.0, "Argon": 8200.0,
"Neon": 20000.0, "Hydrogen": 24000.0, "Helium": 2000.0,
# Gorenflo reference fluid: P_f = 1, so Eq. (8) gives α₀ = 3580 W/m²K.
"ReferenceFluid": 3580.0,
}

# ---------------------------------------------------------------------------
# VDI Heat Atlas 2nd edition — Gorenflo table footnote annotations
#
# Data-quality footnotes on the 2009 α₀,exp column of Table H2.1. They do not
# change the calculation; Gorenflo() warns when a flagged value is used.
#
# Footnote 'd': very few experimental data available.
# Footnote 'e': very high scatter of experimental data.
# (Footnote 'f' only marks values not updated since 1993, and footnote 'i'
#  the helium reference heat flux of 1 kW/m², handled inside Gorenflo().)
# ---------------------------------------------------------------------------
_gorenflo_footnote_d = frozenset({
"92-52-4",   # Biphenyl
"71-36-3",   # n-Butanol
"78-83-1",   # Isobutanol
"754-12-1",  # R1234yf
"75-73-0",   # R14
"306-83-2",  # R123
"7782-44-7", # Oxygen
"7440-37-1", # Argon
"7440-01-9", # Neon
"1333-74-0", # Hydrogen
})

_gorenflo_footnote_e = frozenset({
"71-43-2",   # Benzene
"108-88-3",  # Toluene
"56-23-5",   # R10 (carbon tetrachloride)
})

#Old values, kept as backup in case they are needed
h0_Gorenflow_1993 = {"74-82-8": 7000.0, "74-84-0": 4500.0, "74-98-6": 4000.0,
"106-97-8": 3600.0, "109-66-0": 3400.0, "78-78-4": 2500.0, "110-54-3": 3300.0,
"142-82-5": 3200.0, "71-43-2": 2900.0, "108-88-3": 2800.0, "92-52-4": 2100.0,
"67-56-1": 5400.0, "64-17-5": 4400.0, "71-23-8": 3800.0, "67-63-0": 3000.0,
"71-36-3": 2600.0, "78-83-1": 4500.0, "67-64-1": 3300.0, "75-69-4": 2800.0,
"75-71-8": 4000.0, "75-72-9": 3900.0, "75-63-8": 3500.0, "75-45-6": 3900.0,
"75-46-7": 4400.0, "76-13-1": 2650.0, "76-14-2": 3800.0, "76-15-3": 3200.0,
"811-97-2": 4500.0, "28987-04-4": 3700.0, "431-89-0": 3800.0, "115-25-3": 4200.0,
"74-87-3": 4400.0, "56-23-5": 3200.0, "75-73-0": 4750.0, "7732-18-5": 5600.0,
"7664-41-7": 7000.0, "124-38-9": 5100.0, "2551-62-4": 3700.0, "7782-44-7": 9500.0,
"7727-37-9": 10000.0, "7440-37-1": 8200.0, "7440-01-9": 20000.0, "1333-74-0": 24000.0,
"7440-59-7": 2000.0}


# ---------------------------------------------------------------------------
# Fluid name → CASRN lookup for Gorenflo().
# Primary names match CoolProp fluid identifiers where available.
# Fluids not in CoolProp use standard IUPAC or refrigerant designations.
# Names are case-insensitive inside Gorenflo(); aliases are listed explicitly
# for the most common alternate spellings.
# ---------------------------------------------------------------------------
gorenflo_fluid_aliases = {
# Hydrocarbons — light
'Methane': '74-82-8',
'Ethylene': '74-85-1',        # CoolProp: Ethylene
'Ethene': '74-85-1',
'Ethane': '74-84-0',
'Propylene': '115-07-1',      # CoolProp: Propylene
'Propene': '115-07-1',
'Propane': '74-98-6',
'n-Butane': '106-97-8',
'nButane': '106-97-8',
'IsoButane': '75-28-5',       # CoolProp: IsoButane
'Isobutane': '75-28-5',
'n-Pentane': '109-66-0',
'nPentane': '109-66-0',
'Isopentane': '78-78-4',      # CoolProp: Isopentane
'n-Hexane': '110-54-3',
'nHexane': '110-54-3',
'CycloHexane': '110-82-7',    # CoolProp: CycloHexane
'Cyclohexane': '110-82-7',
'n-Heptane': '142-82-5',
'nHeptane': '142-82-5',
# Aromatics
'Benzene': '71-43-2',
'Toluene': '108-88-3',
'Biphenyl': '92-52-4',        # not in CoolProp; chemical name used
# Alcohols
'Methanol': '67-56-1',
'Ethanol': '64-17-5',
'n-Propanol': '71-23-8',
'nPropanol': '71-23-8',
'1-Propanol': '71-23-8',
'Isopropanol': '67-63-0',     # 2-Propanol; not in CoolProp
'2-Propanol': '67-63-0',
'n-Butanol': '71-36-3',
'nButanol': '71-36-3',
'1-Butanol': '71-36-3',
'IsoButanol': '78-83-1',      # CoolProp: IsoButanol
'Isobutanol': '78-83-1',
'2-Butanol': '78-92-2',       # sec-Butanol; not in CoolProp
'sec-Butanol': '78-92-2',
# Other organics
'Acetaldehyde': '75-07-0',    # not in CoolProp; chemical name used
'Acetone': '67-64-1',         # CoolProp: Acetone
# Inorganic / permanent gases
'CO2': '124-38-9',
'CarbonDioxide': '124-38-9',
'Water': '7732-18-5',
'H2O': '7732-18-5',
'Ammonia': '7664-41-7',
'NH3': '7664-41-7',
'Oxygen': '7782-44-7',
'O2': '7782-44-7',
'Nitrogen': '7727-37-9',
'N2': '7727-37-9',
'Argon': '7440-37-1',
'Ar': '7440-37-1',
'Neon': '7440-01-9',
'Ne': '7440-01-9',
'Hydrogen': '1333-74-0',
'H2': '1333-74-0',
'Helium': '7440-59-7',
'He': '7440-59-7',
'SF6': '2551-62-4',
# HFCs / HCFCs / CFCs (names match CoolProp where available)
'R11': '75-69-4',
'R12': '75-71-8',
'R13': '75-72-9',
'R13B1': '75-63-8',           # bromotrifluoromethane; not in CoolProp
'R14': '75-73-0',             # CF4; CoolProp: R14
'R22': '75-45-6',
'R23': '75-46-7',
'R32': '75-10-5',
'R40': '74-87-3',             # chloromethane; CoolProp: R40
'Chloromethane': '74-87-3',
'R10': '56-23-5',             # carbon tetrachloride; not in CoolProp
'CarbonTetrachloride': '56-23-5',
'R113': '76-13-1',
'R114': '76-14-2',
'R115': '76-15-3',
'R123': '306-83-2',
'R125': '354-33-6',
'R134a': '811-97-2',
'R143a': '420-46-2',
'R152a': '75-37-6',
'R218': '115-25-3',           # octafluoropropane alias (RC318 preferred)
'RC318': '115-25-3',          # octafluorocyclobutane; not in CoolProp
'R227EA': '431-89-0',         # CoolProp: R227EA
'R227ea': '431-89-0',
'R1234yf': '754-12-1',
# 1993-only refrigerant
'R502': '28987-04-4',         # azeotrope R22/R115; not in CoolProp
# Gorenflo reference fluid (P_f,ref = 1, α₀,ref = 3580 W/m²K)
'ReferenceFluid': 'reference',
'Reference': 'reference',
'reference': 'reference',
}

# Reverse mapping: CASRN → preferred display name
gorenflo_casrn_to_name = {
'74-82-8': 'Methane', '74-85-1': 'Ethylene', '74-84-0': 'Ethane',
'115-07-1': 'Propylene', '74-98-6': 'Propane', '106-97-8': 'n-Butane',
'75-28-5': 'IsoButane', '109-66-0': 'n-Pentane', '78-78-4': 'Isopentane',
'110-54-3': 'n-Hexane', '110-82-7': 'CycloHexane', '142-82-5': 'n-Heptane',
'71-43-2': 'Benzene', '108-88-3': 'Toluene', '92-52-4': 'Biphenyl',
'67-56-1': 'Methanol', '64-17-5': 'Ethanol', '71-23-8': 'n-Propanol',
'67-63-0': 'Isopropanol', '71-36-3': 'n-Butanol', '78-83-1': 'IsoButanol',
'78-92-2': '2-Butanol', '75-07-0': 'Acetaldehyde', '67-64-1': 'Acetone',
'124-38-9': 'CO2', '75-46-7': 'R23', '75-10-5': 'R32', '354-33-6': 'R125',
'811-97-2': 'R134a', '420-46-2': 'R143a', '75-37-6': 'R152a',
'754-12-1': 'R1234yf', '431-89-0': 'R227EA', '115-25-3': 'RC318',
'75-73-0': 'R14', '306-83-2': 'R123', '75-69-4': 'R11', '75-71-8': 'R12',
'75-72-9': 'R13', '75-63-8': 'R13B1', '75-45-6': 'R22', '76-13-1': 'R113',
'76-14-2': 'R114', '76-15-3': 'R115', '74-87-3': 'R40', '56-23-5': 'R10',
'2551-62-4': 'SF6', '7732-18-5': 'Water', '7664-41-7': 'Ammonia',
'7782-44-7': 'Oxygen', '7727-37-9': 'Nitrogen', '7440-37-1': 'Argon',
'7440-01-9': 'Neon', '1333-74-0': 'Hydrogen', '7440-59-7': 'Helium',
'28987-04-4': 'R502',
'reference': 'ReferenceFluid',
}

#%% Find relavent HTC, with error checks and warnings
def _gorenflo_data_warning(casrn):
    if casrn in _gorenflo_footnote_d:
        reason = "is based on very few experimental data (footnote d)"
    elif casrn in _gorenflo_footnote_e:
        reason = "is based on data with very high scatter (footnote e)"
    else:
        return
    warnings.warn("Gorenflo (2010): the tabulated h0 for {} (CASRN {}) {}; "
                  "expect larger uncertainty.".format(
                      gorenflo_casrn_to_name.get(casrn, casrn), casrn, reason),
                  UserWarning, stacklevel=3)
        
# Internal CASRN-keyed lookup for Gorenflo() — derived from the human-readable
# h0_VDI_2e (name-keyed) via gorenflo_casrn_to_name (CASRN → name mapping).
# The key 'reference' maps to 'ReferenceFluid' so the reference-fluid path works.
_h0_VDI_2e_by_casrn = {casrn: h0_VDI_2e[name]
                        for casrn, name in gorenflo_casrn_to_name.items()
                        if name in h0_VDI_2e}

# Internal lowercase lookup used inside Gorenflo() for case-insensitive matching
_gorenflo_fluid_aliases_lower = {k.lower(): v for k, v in gorenflo_fluid_aliases.items()}

#Check is a fluid was Given
# Check if a CASRN was given : explicit CASRN takes priority, then fluid name lookup
_casrn_used = CASRN
if _casrn_used is None and fluid is not None:
    _casrn_used = _gorenflo_fluid_aliases_lower.get(fluid.lower())
    if _casrn_used is None:
        raise ValueError("Fluid name '{}' not found in Gorenflo tables. "
            "See gorenflo_fluid_aliases for valid names, or pass CASRN directly.".format(fluid))

# A fluid not in the tables (or the reference fluid) with dPdT and sigma given
# gets h0 from Eq. (8) in the "Unlisted fluid" cell below, so skip the lookup.
use_reference_fluid = (h0 is None and dPdT is not None and sigma is not None
                       and (_casrn_used is None or _casrn_used == "reference"))

if h0 is None and not use_reference_fluid:
    if _casrn_used in _h0_VDI_2e_by_casrn:
        h0 = _h0_VDI_2e_by_casrn[_casrn_used]
    elif _casrn_used in h0_Gorenflow_1993:
        h0 = h0_Gorenflow_1993[_casrn_used]
    elif _casrn_used is None:
        raise ValueError("Provide CASRN, fluid or h0; or, for a fluid not in the "
                         "tables, dPdT and sigma at p* = 0.1 to estimate h0")
    else:
        raise ValueError("Reference heat transfer coefficient not known for: " + str(_casrn_used))
if not h0_given: _gorenflo_data_warning(_casrn_used)



if not 0.0 < Pr < 1.0:
    raise ValueError("Reduced pressure P/Pc must be between 0 and 1")


        
#%% Unlisted fluid - use reference fluid
# estimate h0 with Eq. (8), P_f evaluated at p* = 0.1.
# P_f in (kPa/K)/(mN/m): dPdT [Pa/K]/1E3 and sigma [N/m]*1E3 -> dPdT/(sigma*1E6)
if use_reference_fluid:
    P_f = dPdT/sigma
    P_f0 = 1e6
    F_f = (P_f/P_f0)**0.6
    h0 = 3580.0* F_f

#%% Pressure and Heat Flux Influence

if _casrn_used == "7732-18-5":
    # Water-specific equations, VDI Heat Atlas H2
    n = 0.9 - 0.3*Pr**0.15
    Fp = 1.73*Pr**0.27 + (6.1 + 0.68/(1.0 - Pr))*Pr*Pr
else:
    #All other fluids
    n = 0.95 - 0.3*Pr**0.3
    Fp = 0.7*Pr**0.2 + 4.0*Pr + 1.4*Pr/(1.0 - Pr)
if _casrn_used == "7440-59-7":
    # Helium h0 is given at q0 = 1 kW/m^2 (Table H2.1, footnote i)
    q0 = 1E3

#%% Wall correction: surface roughness x wall-material effusivity
F_wr = (Ra/Ra0)**(2.0/15.0)
F_wm = (eff/eff_Cu)**0.5
F_w = F_wr * F_wm

#%% HTC calculation
if q is not None:
    h = h0*F_w*Fp*(q/q0)**n
elif Te is not None:
    # h = h0*F_w*Fp*(q/q0)^n with q = h*Te  ->  h^(1-n) = h0*F_w*Fp*(Te/q0)^n
    A = h0*F_w*Fp*(Te/q0)**n
    h = A**(1./(1. - n))
    # Solving for h raises every input factor to the power 1/(1-n), so any
    # error in h0, F_w or F(p*) is amplified. Warn the user how much.
    amplification = 1./(1. - n)
    warnings.warn(
        "Gorenflo with specified Te: h scales with h0*F_w*F(p*) raised to "
        "1/(1-n) = {:.1f}, so a 10% error in h0, F_w or F(p*) gives about a "
        "{:.0f}% error in h (and in q = h*Te). Specify q instead of Te where "
        "possible.".format(amplification, 100*(1.1**amplification - 1)),
        UserWarning)
else:
    raise ValueError("Either q or Te is needed for this correlation")

print(f" h = {h:.2f}")


#%% Test of the file

h_gorenflo = h
q_gorenflo = q if q is not None else h*Te

print("\n  3. Gorenflo's Relation:")
print(f" h0 = {h0:.0f} W/m^2·K ({'Eq. (8) estimate' if use_reference_fluid else 'tabulated or given'})")
print(f" n = {n:.3f}")
print(f" Fp = {Fp:.2f}")
print(f" F_wr = {F_wr:.2f}")
print(f" F_wm = {F_wm:.2f}")
print(f" F_w = {F_w:.2f}")
print(f" F_f = {F_f:.2f}")

print(f"     - Heat Transfer Coeff (h)   = {h_gorenflo:.2f} W/m^2·K")
print(f"     - Boiling Heat Flux (q'')   = {q_gorenflo:.2f} W/m^2")