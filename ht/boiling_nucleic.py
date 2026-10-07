"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import warnings
from math import log10

from fluids.constants import g

__all__: list[str] = [
    "Bier",
    "Cooper",
    "Forster_Zuber",
    "Gorenflo",
    "HEDH_Montinsky",
    "HEDH_Taborek",
    "McNelly",
    "Montinsky",
    "Rohsenow",
    "Serth_HEDH",
    "Stephan_Abdelsalam",
    "Zuber",
    "gorenflo_casrn_to_name",
    "gorenflo_fluid_aliases",
    "h0_Gorenflow_1993",
    "h0_VDI_2e",
    "h_nucleic",
    "h_nucleic_all_methods",
    "h_nucleic_methods",
    "qmax_boiling",
    "qmax_boiling_all_methods",
    "qmax_boiling_methods",
]


def Rohsenow(rhol: float, rhog: float, mul: float, kl: float, Cpl: float, Hvap: float, sigma: float, Te: float | None=None, q: float | None=None, Csf: float=0.013,
             n: float=1.7) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = {{\mu }_{L}} \Delta H_{vap} \left[ \frac{g( \rho_L-\rho_v)}
        {\sigma } \right]^{0.5}\left[\frac{C_{p,L}\Delta T_e^{2/3}}{C_{sf}
        \Delta H_{vap} Pr_L^n}\right]^3

    With `q` specified:

    .. math::
        h = \left({{\mu }_{L}} \Delta H_{vap} \left[ \frac{g( \rho_L-\rho_v)}
        {\sigma } \right]^{0.5}\left[\frac{C_{p,L}\Delta T_e^{2/3}}{C_{sf}
        \Delta H_{vap} Pr_L^n}\right]^3\right)^{1/3}q^{2/3}

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    Csf : float
        Rohsenow coefficient specific to fluid and metal [-]
    n : float
        Constant, 1 for water, 1.7 (default) for other fluids usually [-]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No further work is required on this correlation. Multiple sources confirm
    its form and rearrangement.

    Examples
    --------
    h for water at atmospheric pressure on oxidized aluminum.

    >>> Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217,
    ... Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26)
    3723.655267067467

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Rohsenow, Warren M. "A Method of Correlating Heat Transfer Data for
       Surface Boiling of Liquids." Technical Report. Cambridge, Mass. : M.I.T.
       Division of Industrial Cooporation, 1951
    """
    if Te is not None:
        return mul*Hvap*(g*(rhol-rhog)/sigma)**0.5*(Cpl*Te**(2/3.)/Csf/Hvap/(Cpl*mul/kl)**n)**3
    elif q is not None:
        A = mul*Hvap*(g*(rhol-rhog)/sigma)**0.5*(Cpl/Csf/Hvap/(Cpl*mul/kl)**n)**3
        return A**(1/3.)*q**(2/3.)
    else:
        raise ValueError("Either q or Te is needed for this correlation")


def McNelly(rhol: float, rhog: float, kl: float, Cpl: float, Hvap: float, sigma: float, P: float, Te: float | None=None, q: float | None=None) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(0.225\left(\frac{\Delta T_e C_{p,l}}{H_{vap}}\right)^{0.69}
        \left(\frac{P k_L}{\sigma}\right)^{0.31}
        \left(\frac{\rho_L}{\rho_V}-1\right)^{0.33}\right)^{1/0.31}

    With `q` specified:

    .. math::
        h = 0.225\left(\frac{q C_{p,l}}{H_{vap}}\right)^{0.69} \left(\frac{P
        k_L}{\sigma}\right)^{0.31}\left(\frac{\rho_L}{\rho_V}-1\right)^{0.33}

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    P : float
        Saturation pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Further examples for this function are desired.

    Examples
    --------
    Water boiling, with excess temperature of 4.3 K.

    >>> McNelly(Te=4.3, P=101325, Cpl=4180., kl=0.688, sigma=0.0588,
    ... Hvap=2.25E6, rhol=958., rhog=0.597)
    533.8056972951352

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] McNelly M. J.: "A correlation of the rates of heat transfer to n
       ucleate boiling liquids," J. Imp Coll. Chem Eng Soc 7:18, 1953.
    """
    if Te is not None:
        return (0.225*(Te*Cpl/Hvap)**0.69*(P*kl/sigma)**0.31*(rhol/rhog-1.)**0.33
            )**(1./0.31)
    elif q is not None:
        return 0.225*(q*Cpl/Hvap)**0.69*(P*kl/sigma)**0.31*(rhol/rhog-1.)**0.33
    else:
        raise ValueError("Either q or Te is needed for this correlation")


def Forster_Zuber(rhol: float, rhog: float, mul: float, kl: float, Cpl: float, Hvap: float, sigma: float, dPsat: float, Te: float | None=None, q: float | None=None) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = 0.00122\left(\frac{k_L^{0.79} C_{p,l}^{0.45}\rho_L^{0.49}}
        {\sigma^{0.5}\mu_L^{0.29} H_{vap}^{0.24} \rho_V^{0.24}}\right)
        \Delta T_e^{0.24} \Delta P_{sat}^{0.75}

    With `q` specified:

    .. math::
        h = \left[0.00122\left(\frac{k_L^{0.79} C_{p,l}^{0.45}\rho_L^{0.49}}
        {\sigma^{0.5}\mu_L^{0.29} H_{vap}^{0.24} \rho_V^{0.24}}\right) \Delta
        P_{sat}^{0.75} q^{0.24}\right]^{\frac{1}{1.24}}

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    dPsat : float
        Difference in saturation pressure of the fluid at Te and T, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Examples have been found in [1]_ and [3]_ and match exactly.

    Examples
    --------
    Water boiling, with excess temperature of 4.3K from [1]_.

    >>> Forster_Zuber(Te=4.3, dPsat=3906*4.3, Cpl=4180., kl=0.688,
    ... mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    3519.9239897462644

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Forster, H. K., and N. Zuber. "Dynamics of Vapor Bubbles and Boiling
       Heat Transfer." AIChE Journal 1, no. 4 (December 1, 1955): 531-35.
       doi:10.1002/aic.690010425.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    """
    if Te is not None:
        return 0.00122*(kl**0.79*Cpl**0.45*rhol**0.49/sigma**0.5/mul**0.29/Hvap**0.24/rhog**0.24)*Te**0.24*dPsat**0.75
    elif q is not None:
        return (0.00122*(kl**0.79*Cpl**0.45*rhol**0.49/sigma**0.5/mul**0.29/Hvap**0.24/rhog**0.24)*q**0.24*dPsat**0.75)**(1/1.24)
    else:
        raise ValueError("Either q or Te is needed for this correlation")


def Montinsky(P: float, Pc: float, Te: float | None=None, q: float | None=None) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[1.8(P/P_c)^{0.17} +
        4(P/P_c)^{1.2} + 10(P/P_c)^{10}\right]\right)^{1/0.3}

    With `q` specified:

    .. math::
        h = 0.00417P_c^{0.69} q^{0.7}\left[1.8(P/P_c)^{0.17} + 4(P/P_c)^{1.2}
        + 10(P/P_c)^{10}\right]

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Formulas has been found consistent in all cited sources. Examples have
    been found in [1]_ and [3]_.

    The equation for this function is sometimes given with a constant of 3.7E-5
    instead of 0.00417 if critical pressure is not internally
    converted to kPa. [3]_ lists a constant of 3.596E-5.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3K from [1]_.

    >>> Montinsky(P=101325, Pc=22048321, Te=4.3)
    1185.0509770292663

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Mostinsky I. L.: "Application of the rule of corresponding states
       for the calculation of heat transfer and critical heat flux,"
       Teploenergetika 4:66, 1963 English Abstr. Br Chem Eng 8(8):586, 1963
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [4] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    """
    if Te is not None:
        return (0.00417*(Pc/1000.)**0.69*Te**0.7*(1.8*(P/Pc)**0.17 + 4*(P/Pc)**1.2
        +10*(P/Pc)**10))**(1/0.3)
    elif q is not None:
        return (0.00417*(Pc/1000.)**0.69*q**0.7*(1.8*(P/Pc)**0.17 + 4*(P/Pc)**1.2
        +10*(P/Pc)**10))
    else:
        raise ValueError("Either q or Te is needed for this correlation")


_angles_Stephan_Abdelsalam = {"general": 35, "water": 45, "hydrocarbon": 35,
"cryogenic": 1, "refrigerant": 35}

def Stephan_Abdelsalam(rhol: float, rhog: float, mul: float, kl: float, Cpl: float, Hvap: float, sigma: float, Tsat: float, Te: float | None=None,
                       q: float | None=None, kw: float=401.0, rhow: float=8.96, Cpw: float=384.0, angle: float | None=None,
                       correlation: str="general") -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.
    Five variants are possible.

    Either heat flux or excess temperature is required. The forms for `Te` are
    not shown here, but are similar to those of the other functions.

    .. math::
        h = 0.23X_1^{0.674} X_2^{0.35} X_3^{0.371} X_5^{0.297} X_8^{-1.73} k_L/d_B

    .. math::
        X1 = \frac{q D_d}{K_L T_{sat}}

    .. math::
        X2 = \frac{\alpha^2 \rho_L}{\sigma D_d}

    .. math::
        X3 = \frac{C_{p,L} T_{sat} D_d^2}{\alpha^2}

    .. math::
        X4 = \frac{H_{vap} D_d^2}{\alpha^2}

    .. math::
        X5 = \frac{\rho_V}{\rho_L}

    .. math::
        X6 = \frac{C_{p,l} \mu_L}{k_L}

    .. math::
        X7 = \frac{\rho_W C_{p,W} k_W}{\rho_L C_{p,L} k_L}

    .. math::
        X8 = \frac{\rho_L-\rho_V}{\rho_L}

    .. math::
        D_b = 0.0146\theta\sqrt{\frac{2\sigma}{g(\rho_L-\rho_g)}}

    Respectively, the following four correlations are for water, hydrocarbons,
    cryogenic fluids, and refrigerants.

    .. math::
        h = 0.246\times 10^7 X1^{0.673} X4^{-1.58} X3^{1.26}X8^{5.22}k_L/d_B

    .. math::
        h = 0.0546 X5^{0.335} X1^{0.67} X8^{-4.33} X4^{0.248}k_L/d_B

    .. math::
        h = 4.82 X1^{0.624} X7^{0.117} X3^{0.374} X4^{-0.329}X5^{0.257} k_L/d_B

    .. math::
        h = 207 X1^{0.745} X5^{0.581} X6^{0.533} k_L/d_B

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    Tsat : float
        Saturation temperature at operating pressure [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    kw : float, optional
        Thermal conductivity of wall (only for cryogenics) [W/m/K]
    rhow : float, optional
        Density of the wall (only for cryogenics) [kg/m^3]
    Cpw : float, optional
        Heat capacity of wall (only for cryogenics) [J/kg/K]
    angle : float, optional
        Contact angle of bubble with wall [degrees]
    correlation : str, optional
        Any of 'general', 'water', 'hydrocarbon', 'cryogenic', or 'refrigerant'

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    If cryogenic correlation is selected, metal properties are used. Default
    values are the properties of copper at STP.

    The angle is selected automatically if a correlation is selected; if angle
    is provided anyway, the automatic selection is ignored. A IndexError
    exception is raised if the correlation is not in the dictionary
    _angles_Stephan_Abdelsalam.

    Examples
    --------
    Example is from [3]_ and matches.

    >>> Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,
    ... sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, angle=35)
    26722.441071108373

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Stephan, K., and M. Abdelsalam. "Heat-Transfer Correlations for
       Natural Convection Boiling." International Journal of Heat and Mass
       Transfer 23, no. 1 (January 1980): 73-87.
       doi:10.1016/0017-9310(80)90140-4.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    """
    if Te is None and q is None:
        raise ValueError("Either q or Te is needed for this correlation")

    if correlation == "water":
        angle = 45.0
    elif correlation == "cryogenic":
        angle = 1.0
    elif True:
        angle = 35.0

    db = 0.0146*angle*(2*sigma/g/(rhol-rhog))**0.5
    diffusivity_L = kl/rhol/Cpl

    if Te is not None:
        X1 = db/kl/Tsat*Te
    elif q is not None:
        X1 = db/kl/Tsat*q
    X2 = diffusivity_L**2*rhol/sigma/db
    X3 = Hvap*db**2/diffusivity_L**2
    X4 = Hvap*db**2/diffusivity_L**2
    X5 = rhog/rhol
    X6 = Cpl*mul/kl
    X7 = rhow*Cpw*kw/(rhol*Cpl*kl)
    X8 = (rhol-rhog)/rhol

    if correlation == "general":
        if Te is not None:
            h = (0.23*X1**0.674*X2**0.35*X3**0.371*X5**0.297*X8**-1.73*kl/db)**(1/0.326)
        else:
            h = (0.23*X1**0.674*X2**0.35*X3**0.371*X5**0.297*X8**-1.73*kl/db)
    elif correlation == "water":
        if Te is not None:
            h = (0.246E7*X1**0.673*X4**-1.58*X3**1.26*X8**5.22*kl/db)**(1/0.327)
        else:
            h = (0.246E7*X1**0.673*X4**-1.58*X3**1.26*X8**5.22*kl/db)
    elif correlation == "hydrocarbon":
        if Te is not None:
            h = (0.0546*X5**0.335*X1**0.67*X8**-4.33*X4**0.248*kl/db)**(1/0.33)
        else:
            h = (0.0546*X5**0.335*X1**0.67*X8**-4.33*X4**0.248*kl/db)
    elif correlation == "cryogenic":
        if Te is not None:
            h = (4.82*X1**0.624*X7**0.117*X3**0.374*X4**-0.329*X5**0.257*kl/db)**(1/0.376)
        else:
            h = (4.82*X1**0.624*X7**0.117*X3**0.374*X4**-0.329*X5**0.257*kl/db)
    else:
        if Te is not None:
            h = (207*X1**0.745*X5**0.581*X6**0.533*kl/db)**(1/0.255)
        else:
            h = (207*X1**0.745*X5**0.581*X6**0.533*kl/db)
    return h


def HEDH_Taborek(P: float, Pc: float, Te: float | None=None, q: float | None=None) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to Taborek (1986)
    as described in [1]_ and as presented in [2]_. Modification of [3]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[2.1P_r^{0.27} +
        \left(9 + (1-Pr^2)^{-1}\right)P_r^2 \right]\right)^{1/0.3}

    With `q` specified:

    .. math::
        h = 0.00417P_c^{0.69} q^{0.7}\left[2.1P_r^{0.27} + \left(9 + (1-Pr^2
        )^{-1}\right)P_r^2\right]

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Example is from [3]_ and matches to within the error of the algebraic
    manipulation rounding.

    Examples
    --------
    >>> HEDH_Taborek(Te=16.2, P=310.3E3, Pc=2550E3)
    1397.272486525486

    References
    ----------
    .. [1] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [2] Mostinsky I. L.: "Application of the rule of corresponding states
       for the calculation of heat transfer and critical heat flux,"
       Teploenergetika 4:66, 1963 English Abstr. Br Chem Eng 8(8):586, 1963
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    """
    Pr = P/Pc
    if Te is not None:
        return (0.00417*(Pc/1000.)**0.69*Te**0.7*(2.1*Pr**0.27
        + (9 + 1./(1-Pr**2))*Pr**2))**(1/0.3)
    elif q is not None:
        return (0.00417*(Pc/1000.)**0.69*q**0.7*(2.1*Pr**0.27
        + (9 + 1./(1-Pr**2))*Pr**2))
    else:
        raise ValueError("Either q or Te is needed for this correlation")


def Bier(P: float, Pc: float, Te: float | None=None, q: float | None=None) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [1]_ .

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[0.7 + 2P_r\left(4 +
        \frac{1}{1-P_r}\right)  \right]\right)^{1/0.3}

    With `q` specified:

    .. math::
        h = 0.00417P_c^{0.69} \Delta q^{0.7}\left[0.7 + 2P_r\left(4 +
        \frac{1}{1-P_r}\right)  \right]

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No examples of this are known. Seems to give very different results than
    other correlations.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3 K from [1]_.

    >>> Bier(101325., 22048321.0, Te=4.3)
    1290.5349471503353

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    """
    Pr = P/Pc
    if Te is not None:
        return (0.00417*(Pc/1000.)**0.69*Te**0.7*(0.7 + 2.*Pr*(4. + 1./(1.-Pr))))**(1./0.3)
    elif q is not None:
        return 0.00417*(Pc/1000.)**0.69*q**0.7*(0.7 + 2.*Pr*(4. + 1./(1. - Pr)))
    else:
        raise ValueError("Either q or Te is needed for this correlation")


def Cooper(P: float, Pc: float, MW: float, Te: float | None=None, q: float | None=None, Rp: float=1E-6) -> float:
    r"""Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(55\Delta Te^{0.67} \frac{P}{P_c}^{(0.12 - 0.2\log_{10} R_p)}
        (-\log_{10} \frac{P}{P_c})^{-0.55} MW^{-0.5}\right)^{1/0.33}

    With `q` specified:

    .. math::
        h = 55q^{0.67} \frac{P}{P_c}^{(0.12 - 0.2\log_{10} R_p)}(-\log_{10}
        \frac{P}{P_c})^{-0.55} MW^{-0.5}

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    MW : float
        Molecular weight of fluid, [g/mol]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    Rp : float, optional
        Roughness parameter of the surface (1 micrometer default) used by
        `Cooper` method, [m]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Examples 1 and 2 are for water and benzene, from [1]_.
    Roughness parameter is with an old definition. Accordingly, it is
    not used by the h function.
    If unchanged, the roughness parameter's logarithm gives a value of 0.12
    as an exponent of reduced pressure.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3 K from [1]_.

    >>> Cooper(P=101325., Pc=22048321.0, MW=18.02, Te=4.3)
    1558.1435442153575

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] M. G. Cooper, "Saturation and Nucleate Pool Boiling: A Simple
       Correlation," Inst. Chem. Eng. Syrup. Ser. (86/2): 785, 1984.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    """
    Rp*= 1E6
    if Te is not None:
        return (55*Te**0.67*(P/Pc)**(0.12 - 0.2*log10(Rp))*(
             -log10(P/Pc))**-0.55*MW**-0.5)**(1/0.33)
    elif q is not None:
        return (55*q**0.67*(P/Pc)**(0.12 - 0.2*log10(Rp))*(
             -log10(P/Pc))**-0.55*MW**-0.5)
    else:
        raise ValueError("Either q or Te is needed for this correlation")


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
IS_NUMBA = "IS_NUMBA" in globals()
if IS_NUMBA:
    h0_Gorenflow_1993_keys = tuple(h0_Gorenflow_1993.keys())
    h0_Gorenflow_1993_values = tuple(h0_Gorenflow_1993.values())

def Gorenflo(P: float, Pc: float, dPdT: float | None=None,
             sigma: float | None=None,
             q: float | None=None, Te: float | None=None,
             CASRN: str | None=None, fluid: str | None=None,
             h0: float | None=None,
             Ra: float=4E-7, eff: float=35350.0,
             return_CASRN: bool=False):
    r"""Calculates the heat transfer coefficient for nucleate pool boiling
    using the Gorenflo (2010) correlation as presented in the VDI Heat Atlas,
    2nd edition [1]_. The correlation is based on the law of corresponding
    states, with a single fluid-specific reference heat transfer coefficient
    :math:`h_0` and correction factors for reduced pressure, heat flux,
    surface roughness and wall material.

    Either `q` or `Te` must be specified. The reference coefficient `h0` is
    taken from, in order of priority:

    1. the `h0` argument;
    2. the VDI 2nd-edition table `h0_VDI_2e` (looked up by `CASRN` or `fluid`);
    3. the 1993 table `h0_Gorenflow_1993`;
    4. Eq. (8) of [1]_, :math:`h_0 = 3580 P_f^{0.6}`, when no fluid is
       identified (or ``fluid='ReferenceFluid'``) and `dPdT` and `sigma`
       are supplied.

    .. math::
        h = h_0 \cdot F(p^*) \cdot \left(\frac{q}{q_0}\right)^n \cdot F_W

    For all fluids except water:

    .. math::
        F(p^*) = 0.7\,{p^*}^{0.2} + 4\,p^* + \frac{1.4\,p^*}{1-p^*}

    .. math::
        n = 0.95 - 0.3\,{p^*}^{0.3}

    For water:

    .. math::
        F(p^*) = 1.73\,{p^*}^{0.27}
            + \left(6.1 + \frac{0.68}{1-p^*}\right){p^*}^2

    .. math::
        n = 0.9 - 0.3\,{p^*}^{0.15}

    Wall correction for surface roughness and wall material:

    .. math::
        F_W = \left(\frac{R_a}{R_{a,0}}\right)^{2/15}
            \left(\frac{b}{b_{Cu}}\right)^{0.5}

    Estimate of the reference coefficient for fluids not in the table:

    .. math::
        h_0 = 3580 \cdot P_f^{0.6}, \quad
        P_f = \left.\frac{(dP_{sat}/dT)\,[\text{kPa/K}]}
            {\sigma\,[\text{mN/m}]}\right|_{p^* = 0.1}

    Parameters
    ----------
    P : float
        Saturation pressure of the fluid, [Pa]
    Pc : float
        Critical pressure of the fluid, [Pa]
    dPdT : float, optional
        Slope of the saturation pressure curve, :math:`dP_{sat}/dT`,
        evaluated at the reference reduced pressure :math:`p^* = 0.1`
        (not at the operating pressure). Only used to estimate `h0` for
        fluids not in the tables, [Pa/K]
    sigma : float, optional
        Surface tension of the liquid at :math:`p^* = 0.1`. Only used to
        estimate `h0` for fluids not in the tables, [N/m]
    q : float, optional
        Heat flux, [W/m^2]
    Te : float, optional
        Excess wall temperature (wall superheat), [K]
    CASRN : str, optional
        CAS Registry Number of the fluid; used to look up `h0` and to select
        the water-specific equations. Takes priority over `fluid`, [-]
    fluid : str, optional
        Fluid name, case-insensitive; see `gorenflo_fluid_aliases` for the
        recognised names, [-]
    h0 : float, optional
        Reference heat transfer coefficient at :math:`p^* = 0.1`,
        :math:`q_0` and :math:`R_{a,0}`, [W/m^2/K]
    Ra : float, optional
        Arithmetic-mean surface roughness; the VDI reference value is
        0.4 μm, [m]
    eff : float, optional
        Thermal effusivity of the wall material,
        :math:`b = \sqrt{k \rho c_p}`. Defaults to copper,
        [W*s^0.5/m^2/K]
    return_CASRN : bool, optional
        If True, return a ``(h, CASRN)`` tuple instead of just ``h``, [-]

    Returns
    -------
    h : float
        Nucleate pool boiling heat transfer coefficient, [W/m^2/K]
    CASRN : str or None
        CAS Registry Number used for the `h0` lookup; only returned when
        ``return_CASRN=True``. ``None`` if no fluid was identified, [-]

    Notes
    -----
    Reference conditions (VDI Heat Atlas, 2nd ed., Table H2.1):

    * Reference heat flux             :math:`q_0 = 20\,000` W/m² (1000 W/m²
      for helium, which does not pool boil at 20 kW/m²; table footnote i)
    * Reference surface roughness     :math:`R_{a,0} = 0.4` μm
    * Reference reduced pressure      :math:`p^*_0 = 0.1`
    * Reference wall effusivity       :math:`b_{Cu} = 35\,350` W s^0.5/m²/K

    The fluid parameter :math:`P_f` is used only to *estimate* :math:`h_0`
    (the :math:`\alpha_{0,calc}` column of Table H2.1); it is not applied to
    tabulated :math:`h_0` values, which already contain the fluid's
    properties.

    The water equations are selected only when the fluid is identified as
    water through `CASRN` or `fluid`; if `h0` is supplied directly for water,
    also pass ``CASRN='7732-18-5'``.

    A `UserWarning` is issued for fluids whose tabulated :math:`h_0` is
    flagged in Table H2.1 as based on very few experimental data
    (footnote d) or on data with very high scatter (footnote e).

    Examples
    --------
    R134a boiling at 10 bar with a heat flux of 20 kW/m², copper wall:

    >>> Gorenflo(1E6, 4059280., q=2E4, CASRN='811-97-2')
    8282.243199918714

    Stainless-steel wall (eff = 7730 W s^0.5/m²/K) reduces the result:

    >>> Gorenflo(1E6, 4059280., q=2E4, CASRN='811-97-2', eff=7730.)
    3872.960046980562

    A fluid not in the tables: estimate `h0` from the saturation-curve slope
    and surface tension at :math:`p^* = 0.1` (R134a properties used here, for
    which Table H2.1 lists :math:`\alpha_{0,calc} = 4.26` kW/m²/K):

    >>> Gorenflo(0.1*4059280., 4059280., dPdT=13630., sigma=10.226e-3, q=2E4)
    4241.803323721339

    The following examples verify the implementation against [2]_, in which
    the author of [1]_ works through the method. They are given in kW/m²/K.

    Fig. 1 of [2]_ shows i-Butane (:math:`p_c` = 36.29 bar,
    :math:`\alpha_0` = 3.7 kW/m²/K) boiling at :math:`q_0` for several reduced
    pressures. The values read from the log-scale plot are about 0.87, 1.9,
    8.7 and 25 kW/m²/K at :math:`p^*` = 0.003, 0.03, 0.3 and 0.7; these match
    to within 0.6%, the precision of reading the graph:

    >>> Pc = 36.29e5
    >>> [round(Gorenflo(p*Pc, Pc, q=2E4, CASRN='75-28-5')/1000, 2)
    ...  for p in (0.003, 0.03, 0.3, 0.7)]
    [0.87, 1.89, 8.7, 24.86]

    Eq. (6) of [2]_ (Eq. (8) of [1]_) gives :math:`\alpha_0` = 3.76 kW/m²/K for
    R236fa (:math:`P_f` = 1.087 (μm K)⁻¹) and 2.28 kW/m²/K for hexadecane
    (:math:`P_f` = 0.47 (μm K)⁻¹). Here `dPdT` and `sigma` are chosen to give
    those :math:`P_f` values; the results match after allowing for
    :math:`F(p^* = 0.1)` = 0.997. The :math:`\alpha_{0,calc}` column of
    Table H2.1 in [1]_ (e.g. 7.13 for methane, 4.26 for R134a, 3.87 for
    R1234yf) is matched in the same way to within 0.2%:

    >>> round(Gorenflo(0.1*Pc, Pc, dPdT=10870., sigma=0.01, q=2E4)/1000, 2)
    3.75
    >>> round(Gorenflo(0.1*Pc, Pc, dPdT=4700., sigma=0.01, q=2E4)/1000, 2)
    2.27

    References
    ----------
    .. [1] Gorenflo, D. and Kenning, D., "H2 Pool Boiling", in VDI Heat Atlas,
       2nd Edition, Springer, Berlin, 2010, pp. 757-792.
    .. [2] Gorenflo, D., Baumhögger, E., Herres, G. and Kotthoff, S.,
       "Prediction Methods for Pool Boiling Heat Transfer: A State-of-the-Art
       Review." International Journal of Refrigeration 43 (2014): 203-226.
    """
    

    
    Pr = P/Pc 
    if not 0.0 < Pr < 1.0:
        raise ValueError("Reduced pressure P/Pc must be between 0 and 1")
    Ra0 = 0.4E-6
    q0 = 2E4
    eff_Cu = 35350.0  # W s^0.5 m^-2 K^-1, thermal effusivity of copper
    h0_given = h0 is not None
    # Resolve CASRN: explicit CASRN takes priority, then fluid name lookup
    _casrn_used = CASRN
    if _casrn_used is None and fluid is not None:
        _casrn_used = _gorenflo_fluid_aliases_lower.get(fluid.lower())
        if _casrn_used is None:
            raise ValueError("Fluid name '{}' not found in Gorenflo tables. "
                "See gorenflo_fluid_aliases for valid names, or pass CASRN directly.".format(fluid))
    # Unlisted fluid: estimate h0 with Eq. (8), P_f evaluated at p* = 0.1.
    # P_f in (kPa/K)/(mN/m): dPdT [Pa/K]/1E3 and sigma [N/m]*1E3 -> dPdT/(sigma*1E6)
    if (h0 is None and dPdT is not None and sigma is not None
            and (_casrn_used is None or _casrn_used == "reference")):
        h0 = 3580.0*(dPdT/(sigma*1.0E6))**0.6
    if h0 is None: # NUMBA: DELETE
        if _casrn_used in _h0_VDI_2e_by_casrn:
            h0 = _h0_VDI_2e_by_casrn[_casrn_used]
        elif _casrn_used in h0_Gorenflow_1993:
            h0 = h0_Gorenflow_1993[_casrn_used]
        elif _casrn_used is None:
            raise ValueError("Provide CASRN, fluid or h0; or, for a fluid not in the "
                             "tables, dPdT and sigma at p* = 0.1 to estimate h0")
        else:
            raise ValueError("Reference heat transfer coefficient not known for: " + str(_casrn_used))
    if h0 is None:
        try:
            h0 = _h0_VDI_2e_by_casrn_values[_h0_VDI_2e_by_casrn_keys.index(_casrn_used)]
        except:
            try:
                h0 = h0_Gorenflow_1993_values[h0_Gorenflow_1993_keys.index(_casrn_used)]
            except:
                raise ValueError("Reference heat transfer coefficient not known for: " + str(_casrn_used))
    if not h0_given: _gorenflo_data_warning(_casrn_used) # NUMBA: DELETE
    if _casrn_used == "7732-18-5":
        # Water-specific equations, VDI Heat Atlas H2
        n = 0.9 - 0.3*Pr**0.15
        Fp = 1.73*Pr**0.27 + (6.1 + 0.68/(1.0 - Pr))*Pr*Pr
    else:
        n = 0.95 - 0.3*Pr**0.3
        Fp = 0.7*Pr**0.2 + 4.0*Pr + 1.4*Pr/(1.0 - Pr)
    if _casrn_used == "7440-59-7":
        # Helium h0 is given at q0 = 1 kW/m^2 (Table H2.1, footnote i)
        q0 = 1E3
    # Wall correction: surface roughness x wall-material effusivity
    F_w = (Ra/Ra0)**(2.0/15.0)*(eff/eff_Cu)**0.5
    if q is not None:
        h = h0*F_w*Fp*(q/q0)**n
    elif Te is not None:
        # h = h0*F_w*Fp*(q/q0)^n with q = h*Te  ->  h^(1-n) = h0*F_w*Fp*(Te/q0)^n
        A = h0*F_w*Fp*(Te/q0)**n
        h = A**(1./(1. - n))
    else:
        raise ValueError("Either q or Te is needed for this correlation")
    if return_CASRN:
        return h, _casrn_used
    return h


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

# Internal CASRN-keyed lookup for Gorenflo() — derived from the human-readable
# h0_VDI_2e (name-keyed) via gorenflo_casrn_to_name (CASRN → name mapping).
# The key 'reference' maps to 'ReferenceFluid' so the reference-fluid path works.
_h0_VDI_2e_by_casrn = {casrn: h0_VDI_2e[name]
                        for casrn, name in gorenflo_casrn_to_name.items()
                        if name in h0_VDI_2e}
if IS_NUMBA:
    _h0_VDI_2e_by_casrn_keys = tuple(_h0_VDI_2e_by_casrn.keys())
    _h0_VDI_2e_by_casrn_values = tuple(_h0_VDI_2e_by_casrn.values())

# Internal lowercase lookup used inside Gorenflo() for case-insensitive matching
_gorenflo_fluid_aliases_lower = {k.lower(): v for k, v in gorenflo_fluid_aliases.items()}


cryogenics = {"132259-10-0": "Air", "7440-37-1": "Argon", "630-08-0":
"carbon monoxide", "7782-39-0": "deuterium", "7782-41-4": "fluorine",
"7440-59-7": "helium", "1333-74-0": "hydrogen", "7439-90-9": "krypton",
"74-82-8": "methane", "7440-01-9": "neon", "7727-37-9": "nitrogen",
"7782-44-7": "oxygen", "7440-63-3": "xenon"}

h_nucleic_all_methods = ["Stephan-Abdelsalam", "Stephan-Abdelsalam water",
                     "Stephan-Abdelsalam cryogenic", "HEDH-Taborek",
                     "Forster-Zuber", "Rohsenow", "Cooper", "Bier",
                     "Montinsky", "McNelly", "Gorenflo (2010)"]

def h_nucleic_methods(Te: float | None=None, Tsat: float | None=None, P: float | None=None, dPsat: float | None=None, dPdT: float | None=None, Cpl: float | None=None,
          kl: float | None=None, mul: float | None=None, rhol: float | None=None, sigma: float | None=None, Hvap: float | None=None, rhog: float | None=None,
          MW: float | None=None, Pc: float | None=None, CAS: str | None=None, check_ranges: bool=False) -> list[str]:
    r"""This function returns the names of correlations for nucleate boiling
    heat flux.

    Parameters
    ----------
    Te : float, optional
        Excess wall temperature, [K]
    Tsat : float, optional
        Saturation temperature at operating pressure [Pa]
    P : float, optional
        Saturation pressure of fluid, [Pa]
    dPsat : float, optional
        Difference in saturation pressure of the fluid at Te and T, [Pa]
    Cpl : float, optional
        Heat capacity of liquid [J/kg/K]
    kl : float, optional
        Thermal conductivity of liquid [W/m/K]
    mul : float, optional
        Viscosity of liquid [Pa*s]
    rhol : float, optional
        Density of the liquid [kg/m^3]
    sigma : float, optional
        Surface tension of liquid [N/m]
    Hvap : float, optional
        Heat of vaporization of the fluid at P, [J/kg]
    rhog : float, optional
        Density of the produced gas [kg/m^3]
    MW : float, optional
        Molecular weight of fluid, [g/mol]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    CAS : str, optional
        CAS of fluid
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `h` with the given inputs

    Examples
    --------
    >>> h_nucleic_methods(P=3E5, Pc=22048320., Te=4.0, CAS='7732-18-5')
    ['Gorenflo (2010)', 'HEDH-Taborek', 'Bier', 'Montinsky']
    """
    methods = []
    if P is not None and Pc is not None and CAS is not None:
        if CAS in _h0_VDI_2e_by_casrn or CAS in h0_Gorenflow_1993: # numba: delete
#        if CAS in _h0_VDI_2e_by_casrn_keys or CAS in h0_Gorenflow_1993_keys: # numba: uncomment
            methods.append("Gorenflo (2010)")
    if (Te is not None and Tsat is not None and Cpl is not None and kl is not None
        and mul is not None and sigma is not None and Hvap is not None
        and rhol is not None and rhog is not None):
        if CAS is not None and CAS == "7732-18-5":
            methods.append("Stephan-Abdelsalam water")
        if CAS is not None and CAS in cryogenics:
            methods.append("Stephan-Abdelsalam cryogenic")
        methods.append("Stephan-Abdelsalam")
    if Te is not None and P is not None and Pc is not None:
        methods.append("HEDH-Taborek")
    if (Te is not None and dPsat is not None and Cpl is not None and kl is not None
        and mul is not None and sigma is not None and Hvap is not None
        and rhol is not None and rhog is not None):
        methods.append("Forster-Zuber")
    if (Te is not None and Cpl is not None and kl is not None and mul is not None
        and sigma is not None and Hvap is not None and rhol is not None
        and rhog is not None):
        methods.append("Rohsenow")
    if MW is not None and Te is not None and P is not None and Pc is not None:
        methods.append("Cooper")
    if Te is not None and P is not None and Pc is not None:
        methods.extend(["Bier", "Montinsky"])
    if (Te is not None and P is not None and Cpl is not None and kl is not None
        and sigma is not None and Hvap is not None and rhol is not None
        and rhog is not None):
        methods.append("McNelly")
    return methods


def h_nucleic(Te: float | None=None, q: float | None=None, Tsat: float | None=None, P: float | None=None, dPsat: float | None=None, dPdT: float | None=None,
              Cpl: float | None=None, kl: float | None=None, mul: float | None=None, rhol: float | None=None, sigma: float | None=None, Hvap: float | None=None,
              rhog: float | None=None, MW: float | None=None, Pc: float | None=None, Csf: float=0.013, n: float=1.7, kw: float=401.0, rhow: float=8.96,
              Cpw: float=384.0, angle: float=35.0, Rp: float=1e-6, Ra: float=0.4e-6, h0: float | None=None,
              CAS: str | None=None, Method: str | None=None) -> float:
    r"""This function handles the calculation of nucleate boiling
    heat flux and chooses the best method for performing the calculation
    based on the provided information.

    One of `Te` and `q` are always required.

    Parameters
    ----------
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    Tsat : float, optional
        Saturation temperature at operating pressure [Pa]
    P : float, optional
        Saturation pressure of fluid, [Pa]
    dPsat : float, optional
        Difference in saturation pressure of the fluid at Te and T, [Pa]
    Cpl : float, optional
        Heat capacity of liquid [J/kg/K]
    kl : float, optional
        Thermal conductivity of liquid [W/m/K]
    mul : float, optional
        Viscosity of liquid [Pa*s]
    rhol : float, optional
        Density of the liquid [kg/m^3]
    sigma : float, optional
        Surface tension of liquid [N/m]
    Hvap : float, optional
        Heat of vaporization of the fluid at P, [J/kg]
    rhog : float, optional
        Density of the produced gas [kg/m^3]
    MW : float, optional
        Molecular weight of fluid, [g/mol]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    Csf : float, optional
        Rohsenow coefficient specific to fluid and metal [-]
    n : float, optional
        Rohsenow constant, 1 for water, 1.7 (default) for other fluids usually [-]
    kw : float, optional
        Thermal conductivity of wall (only for cryogenics) [W/m/K]
    rhow : float, optional
        Density of the wall (only for cryogenics) [kg/m^3]
    Cpw : float, optional
        Heat capacity of wall (only for cryogenics) [J/kg/K]
    angle : float, optional
        Contact angle of bubble with wall [degrees]
    Rp : float, optional
        Roughness parameter of the surface (1 micrometer default) used by
        `Cooper` method, [m]
    Ra : float, optional
        Roughness parameter of the surface (0.4 micrometer default) for
        Gorenflo method, [m]
    h0 : float
        Reference heat transfer coefficient for Gorenflo method, [W/m^2/K]
    CAS : str, optional
        CAS of fluid

    Returns
    -------
    h : float
        Nucleate boiling heat flux [W/m^2]

    Other Parameters
    ----------------
    Method : string, optional
        The name of the method to use; one of ['Gorenflo (2010)',
        'Stephan-Abdelsalam water', 'Stephan-Abdelsalam cryogenic',
        'Stephan-Abdelsalam', 'HEDH-Taborek', 'Forster-Zuber', 'Rohsenow',
        'Cooper', 'Bier', 'Montinsky', 'McNelly']

    Notes
    -----
    The methods Stephan-Abdelsalam, Cooper, and Gorenflo all take other
    arguments as well such as surface roughness or the thermal properties of
    the wall material. See them for their documentation. These parameters
    can also be passed as keyword arguments.

    Examples
    --------
    Water, known excess temperature of 4.9 K, Rohsenow method

    Water, known excess temperature of 4.9 K, Rohsenow method

    >>> h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217,
    ... Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26,
    ... Method='Rohsenow')
    3723.655267067467
    """
    if Method is None:
        methods = h_nucleic_methods(Te=Te, Tsat=Tsat, P=P, dPsat=dPsat, dPdT=dPdT, Cpl=Cpl,
              kl=kl, mul=mul, rhol=rhol, sigma=sigma, Hvap=Hvap, rhog=rhog,
              MW=MW, Pc=Pc, CAS=CAS)
        if not methods:
            raise ValueError("Insufficient property data for any method.")
        Method = methods[0]

    if Method == "Stephan-Abdelsalam"and Tsat is not None:
        return Stephan_Abdelsalam(Te=Te, q=q, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               correlation="general",
                               kw=kw, rhow=rhow, Cpw=Cpw, angle=angle)
    elif Method == "Stephan-Abdelsalam water" and Tsat is not None:
        return Stephan_Abdelsalam(Te=Te, q=q, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               correlation="water",
                               kw=kw, rhow=rhow, Cpw=Cpw, angle=angle)
    elif Method == "Stephan-Abdelsalam cryogenic" and Tsat is not None:
        return Stephan_Abdelsalam(Te=Te, q=q, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               correlation="cryogenic",
                               kw=kw, rhow=rhow, Cpw=Cpw, angle=angle)
    elif Method == "HEDH-Taborek" and P is not None and Pc is not None:
        return HEDH_Taborek(Te=Te, q=q, P=P, Pc=Pc)
    elif Method == "Forster-Zuber" and dPsat is not None:
        return Forster_Zuber(Te=Te, q=q, dPsat=dPsat, Cpl=Cpl, kl=kl, mul=mul,
                          sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method == "Rohsenow":
        return Rohsenow(Te=Te, q=q, Cpl=Cpl, kl=kl, mul=mul, sigma=sigma, Hvap=Hvap,
                     rhol=rhol, rhog=rhog, Csf=Csf, n=n)
    elif Method == "Cooper":
        return Cooper(Te=Te, q=q, P=P, Pc=Pc, MW=MW, Rp=Rp)
    elif Method == "Bier" and P is not None and Pc is not None:
        return Bier(Te=Te, q=q, P=P, Pc=Pc)
    elif Method == "Montinsky" and P is not None and Pc is not None:
        return Montinsky(Te=Te, q=q, P=P, Pc=Pc)
    elif Method == "McNelly":
        return McNelly(Te=Te, q=q, P=P, Cpl=Cpl, kl=kl, sigma=sigma, Hvap=Hvap,
                    rhol=rhol, rhog=rhog)

    elif Method == "Gorenflo (2010)":
        # dPdT and sigma are not passed: Gorenflo needs them at p* = 0.1, not at P
        return Gorenflo(P=P, Pc=Pc, q=q, Te=Te, CASRN=CAS, h0=h0, Ra=Ra)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")


### Critical Heat Flux


def Zuber(sigma: float, Hvap: float, rhol: float, rhog: float, K: float=0.18) -> float:
    r"""Calculates critical heat flux for nucleic boiling of a flat plate
    or other shape as presented in various sources.
    K = pi/24 is believed to be the original [1]_ value for K, but 0.149 is
    now more widely used, a value claimed to be from [2]_ according to [5]_.
    Cao [4]_ lists a value of 0.18 for K. The Wolverine Tube data book also
    lists a value of 0.18, and so it is the default.

    .. math::
        q_c = {KH}_{vap} \rho_g^{0.5}\left[\sigma g (\rho_L-\rho_g)\right]^{0.25}

    Parameters
    ----------
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    K : float
        Constant []

    Returns
    -------
    q : float
        Critical heat flux [W/m^2]

    Notes
    -----
    No further work is required on this correlation. Multiple sources confirm
    its form.

    Examples
    --------
    Example from [3]_

    >>> Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, K=0.149)
    444307.22304342285
    >>> Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, K=0.18)
    536746.9808578263

    References
    ----------
    .. [1] Zuber N. "On the stability of boiling heat transfer". Trans ASME 1958
        80:711-20.
    .. [2] Lienhard, J.H., and Dhir, V.K., 1973, Extended Hydrodynamic Theory
       of the Peak and Minimum Heat Fluxes, NASA CR-2270.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [4] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [5] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer, 7E.Mason, OH: Cengage Learning, 2010.
    """
    return K*Hvap*rhog**0.5*(g*sigma*(rhol-rhog))**0.25


def Serth_HEDH(D: float, sigma: float, Hvap: float, rhol: float, rhog: float) -> float:
    r"""Calculates critical heat flux for nucleic boiling of a tube bundle
    according to [2]_, citing [3]_, and using [1]_ as the original form.

    .. math::
        q_c = KH_{vap} \rho_g^{0.5}\left[\sigma g (\rho_L-\rho_g)\right]^{0.25}

    .. math::
        K = 0.123 (R^*)^{-0.25} \text{ for 0.12 < R* < 1.17}

    .. math::
        K = 0.118

    .. math::
        R^* = \frac{D}{2} \left[\frac{g(\rho_L-\rho_G)}{\sigma}\right]^{0.5}

    Parameters
    ----------
    D : float
        Diameter of tubes [m]
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at T, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]

    Returns
    -------
    q : float
        Critical heat flux [W/m^2]

    Notes
    -----
    A further source for this would be nice.

    Examples
    --------
    >>> Serth_HEDH(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    351867.46522901946

    References
    ----------
    .. [1] Zuber N. "On the stability of boiling heat transfer". Trans ASME
       1958 80:711-20.
    .. [2] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [3] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    """
    R = D/2*(g*(rhol-rhog)/sigma)**0.5
    if 0.12 <= R  <= 1.17:
        K = 0.125*R**-0.25
    else:
        K = 0.118
    return K*Hvap*rhog**0.5*(g*sigma*(rhol-rhog))**0.25


def HEDH_Montinsky(P: float, Pc: float) -> float:
    r"""Calculates critical heat flux
    in the nucleate boiling regime according to [3]_ as presented in [1]_,
    using an expression modified from [2]_.

    .. math::
        q_c = 367 P_cP_r^{0.35}(1-P_r)^{0.9}

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]

    Returns
    -------
    q : float
        Critical heat flux [W/m^2]

    Notes
    -----
    No further work is required.
    Units of Pc are kPa internally.

    Examples
    --------
    Example is from [3]_ and matches to within the error of the algebraic
    manipulation rounding.

    >>> HEDH_Montinsky(P=310.3E3, Pc=2550E3)
    398405.66545181436

    References
    ----------
    .. [1] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [2] Mostinsky I. L.: "Application of the rule of corresponding states
       for the calculation of heat transfer and critical heat flux,"
       Teploenergetika 4:66, 1963 English Abstr. Br Chem Eng 8(8):586, 1963
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    """
    Pr = P/Pc
    return 367*(Pc/1000.)*Pr**0.35*(1-Pr)**0.9


qmax_boiling_all_methods = ["Serth-HEDH", "Zuber", "HEDH-Montinsky"]

def qmax_boiling_methods(rhol: int | None=None, rhog: float | None=None, sigma: float | None=None, Hvap: float | None=None, D: float | None=None,
                         P: float | None=None, Pc: float | None=None, check_ranges: bool=False) -> list[str]:
    r"""This function returns a list of methods names which can be used to
    calculate nucleate boiling critical heat flux.
    Preferred methods are 'Serth-HEDH' when a tube diameter is specified,
    and 'Zuber' otherwise.

    Parameters
    ----------
    rhol : float, optional
        Density of the liquid [kg/m^3]
    rhog : float, optional
        Density of the produced gas [kg/m^3]
    sigma : float, optional
        Surface tension of liquid [N/m]
    Hvap : float, optional
        Heat of vaporization of the fluid at T, [J/kg]
    D : float, optional
        Diameter of tubes [m]
    P : float, optional
        Saturation pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    check_ranges : bool, optional
        Added for Future use only

    Returns
    -------
    methods : list
        List of methods which can be used to calculate qmax with the given inputs

    Examples
    --------
    >>> qmax_boiling_methods(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    ['Serth-HEDH', 'Zuber']
    """
    methods = []
    if (sigma is not None and Hvap is not None and rhol is not None
        and rhog is not None and D is not None):
        methods.append("Serth-HEDH")
    if (sigma is not None and Hvap is not None and rhol is not None
        and rhog is not None):
        methods.append("Zuber")
    if P is not None and Pc is not None:
        methods.append("HEDH-Montinsky")
    return methods


def qmax_boiling(rhol: int | None=None, rhog: float | None=None, sigma: float | None=None, Hvap: float | None=None, D: float | None=None, P: float | None=None,
                 Pc: float | None=None, Method: str | None=None) -> float:
    r"""This function handles the calculation of nucleate boiling critical
    heat flux and chooses the best method for performing the calculation.

    Preferred methods are 'Serth-HEDH' when a tube diameter is specified,
    and 'Zuber' otherwise.

    Parameters
    ----------
    rhol : float, optional
        Density of the liquid [kg/m^3]
    rhog : float, optional
        Density of the produced gas [kg/m^3]
    sigma : float, optional
        Surface tension of liquid [N/m]
    Hvap : float, optional
        Heat of vaporization of the fluid at T, [J/kg]
    D : float, optional
        Diameter of tubes [m]
    P : float, optional
        Saturation pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]

    Returns
    -------
    q : float
        Nucleate boiling critical heat flux [W/m^2]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use; one of ('Serth-HEDH', 'Zuber',
        or 'HEDH-Montinsky')

    Examples
    --------
    >>> qmax_boiling(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    351867.46522901946
    """
    if Method is None:
        if (sigma is not None and Hvap is not None and rhol is not None
            and rhog is not None and D is not None):
            Method2 = "Serth-HEDH"
        elif (sigma is not None and Hvap is not None and rhol is not None
            and rhog is not None):
            Method2 = "Zuber"
        elif P is not None and Pc is not None:
            Method2 = "HEDH-Montinsky"
        else:
            raise ValueError("Insufficient property or geometry data for any "
                            "method.")
    else:
        Method2 = Method
    if Method2 == "Serth-HEDH":
        return Serth_HEDH(D=D, sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method2 == "Zuber":
        return Zuber(sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method2 == "HEDH-Montinsky":
        return HEDH_Montinsky(P=P, Pc=Pc)
    else:
        raise ValueError("Correlation name not recognized; options are "
                        "'Serth-HEDH', 'Zuber' and 'HEDH-Montinsky'")
