'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''

from math import cos, exp, log, pi, radians, sin
import warnings

from fluids.numerics import bisplev, horner, implementation_optimize_tck, secant

__all__ = ['Nu_Nusselt_Rayleigh_Holling_Herwig', 'Nu_Nusselt_Rayleigh_Probert',
           'Nu_Nusselt_Rayleigh_Hollands',
           'Rac_Nusselt_Rayleigh', 'Rac_Nusselt_Rayleigh_disk',
           'Nu_Nusselt_vertical_Thess',
           'Nu_vertical_helical_coil_Ali',
           'Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan',
           'Q_enclosure_natural',
           'k_eff_enclosure',
           'Nu_horizontal_enclosure_Jakob',
           'Nu_horizontal_enclosure_Globe_Dropkin',
           'Nu_horizontal_enclosure_Hollands',
           'Nu_inclined_enclosure_Hollands',
           'critical_angle_inclined_enclosure',
           'Nu_inclined_enclosure_Catton',
           'Nu_inclined_enclosure_Arnold',
           'Nu_vertical_enclosure_Berkovsky_Polevikov_1',
           'Nu_vertical_enclosure_Berkovsky_Polevikov_2',
           'Nu_vertical_enclosure_MacGregor_Emery_laminar',
           'Nu_vertical_enclosure_MacGregor_Emery_turbulent',
           'Nu_vertical_enclosure_Cengel',
           'F_concentric_cylinders',
           'k_eff_concentric_cylinders_Raithby_Hollands',
           'Q_concentric_cylinders_natural',
           'F_concentric_spheres',
           'k_eff_concentric_spheres_Raithby_Hollands',
           'Q_concentric_spheres_natural',
           'emissivity_effective_parallel_plates',
           ]

__numba_additional_funcs__ = ['Nu_Nusselt_Rayleigh_Holling_Herwig_err']


def Nu_Nusselt_Rayleigh_Holling_Herwig_err(Nu, Ra, Ra_third, D2):
    err = Ra_third*(0.1/2.0*log(1.0/16.0*Ra*Nu) + D2)**(-4.0/3.0) - Nu
    return err


def Nu_Nusselt_Rayleigh_Holling_Herwig(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical flat horizontal plates. The height between the plates is infinite, and
    one of the other dimensions of the plates is much larger than the other.

    This correlation is for the horizontal plate Rayleigh-Benard classic heat
    transfer problem, not for real finite geometry plates.

    This model is a non-linear equation which is solved numerically.
    The model can calculate `Nu` for `Ra` ranges between 350 and larger
    numbers; [1]_ recommends :math:`10^{5} < Ra < 10^{15}`.

    .. math::
        \text{Nu} = \frac{{Ra}^{1/3}}{[0.05\ln(\frac{0.078}{16}{Ra}^{1.323})
        + 2D]^{4/3}}

    .. math::
        D = -\frac{14.94}{{Ra}^{0.25}} + 3.43

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height between the two plates, [-]

    Notes
    -----
    A range of calculated values are provided in [1]_; they all match the
    results of this function. This model is recommended in [2]_.

    For :math:`Ra < 1708`, `Nu` = 1; for cases not assited by `buoyancy`,
    `Nu` is also 1.

    No success has been found finding an analytical solution in the major CAS
    packages, but the nonlinear function is in fact a function of one variable;
    this means a pade or chebyshev expansion could be performed.


    Examples
    --------
    >>> Nu_Nusselt_Rayleigh_Holling_Herwig(5.54, 3.21e8, buoyancy=True)
    77.54656801896913

    References
    ----------
    .. [1] Hölling, M., and H. Herwig. "Asymptotic Analysis of Heat Transfer in
       Turbulent Rayleigh-Bénard Convection." International Journal of Heat and
       Mass Transfer 49, no. 5 (March 1, 2006): 1129-36.
       https://doi.org/10.1016/j.ijheatmasstransfer.2005.09.002.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if not buoyancy:
        return 1.0
    Rac = 1708 # Constant

    Ra = Gr*Pr
    if Ra < Rac:
        return 1.0

    Ra_third = Ra**(1.0/3.0)
    D2 = 2.0*(-14.94*Ra**-0.25 + 3.43)
    Nu_guess = Ra_third*(0.1/2.0*log(.078/16.0*Ra**1.323) + D2)**(-4.0/3.0)
    return secant(Nu_Nusselt_Rayleigh_Holling_Herwig_err, Nu_guess, args=(Ra, Ra_third, D2))


def Nu_Nusselt_Rayleigh_Probert(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical flat plates. The height between the plates is infinite, and
    one of the other dimensions of the plates is much larger than the other.

    This correlation is for the horizontal plate Rayleigh-Benard classic heat
    transfer problem, not for real finite geometry plates.

    Two sets of equations are used.

    For the laminar regime :math:`1708 < \text{Ra} \le 2.2\times 10^{4}`:

    .. math::
        \text{Nu} = 0.208(\text{Ra})^{0.25}

    For the turbulent regime :math:`2.2\times 10^{4} < \text{Ra}`:

    .. math::
        \text{Nu} = 0.092(\text{Ra})^{1/3}

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height between the two plates, [-]

    Notes
    -----
    This model is recommended in [2]_ as a rough model.

    For :math:`Ra < 1708`, `Nu` = 1; for cases not assited by `buoyancy`,
    `Nu` is also 1.

    Examples
    --------
    >>> Nu_Nusselt_Rayleigh_Probert(5.54, 3.21e8, buoyancy=True)
    111.46181048289132

    References
    ----------
    .. [1] Probert, SD, RG Brooks, and M Dixon. "Heat Transfer across
       Rectangular Cavities." CHEMICAL AND PROCESS ENGINEERING, 1970, 35.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if not buoyancy:
        return 1.0
    Rac = 1708 # Constant

    Ra = Gr*Pr
    if Ra < Rac:
        return 1.0
    elif Ra < 2.2e4:
        return 0.208*Ra**0.25
    else:
        return 0.092*Ra**(1.0/3.0)


def Nu_Nusselt_Rayleigh_Hollands(Pr, Gr, buoyancy=True, Rac=1708):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical flat horizontal plates using the Hollands [1]_ correlation recommended
    in [2]_. This correlation supports different aspect ratios,
    so the plates can be real, finite objects and have their heat transfer
    accurately modeled. The influence comes from the `Rac` term, which should
    be calculated separately, using `Rac_Nusselt_Rayleigh` or
    `Rac_Nusselt_Rayleigh_disk`.

    .. math::
        \text{Nu} = 1 + \left[1 - \frac{1708}{\text{Ra}} \right]^*
        \left[k_1 + 2 \left(\frac{\text{Ra}^{1/3}}{k_2} \right)^{1
        - \ln({\text{Ra}}^{1/5}/k_2)} \right]^*
        + \left[\left(\frac{\text{Ra}}{5803}\right)^{1/3} - 1\right]^*

    .. math::
        k_1 = \frac{1.44}{1 + 0.018/{Pr} + 0.00136/{Pr}^2}

    .. math::
        k_2 = 75\exp(1.5\text{Pr}^{-0.5})

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]
    Rac : float, optional
        Critical Rayleigh number, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height between the two plates, [-]

    Notes
    -----
    For :math:`Ra < {Ra}_c`, `Nu` = 1; for cases not assited by `buoyancy`,
    `Nu` is also 1.

    Examples
    --------
    >>> Nu_Nusselt_Rayleigh_Hollands(5.54, 3.21e8, buoyancy=True)
    69.02668649510

    Plates - 1 m height, 2 m long, 0.2 m long vs a 1 m^3 cube

    >>> Nu_Nusselt_Rayleigh_Hollands(.7, 3.21e6, buoyancy=True, Rac=Rac_Nusselt_Rayleigh(H=1, L=2, W=.2, insulated=False))
    4.666249131876

    >>> Nu_Nusselt_Rayleigh_Hollands(.7, 3.21e6, buoyancy=True, Rac=Rac_Nusselt_Rayleigh(H=1, L=1, W=1, insulated=False))
    8.786362614129

    References
    ----------
    .. [1] Hollands, K. G. T. "Multi-Prandtl Number Correlation Equations for
       Natural Convection in Layers and Enclosures." International Journal of
       Heat and Mass Transfer 27, no. 3 (March 1, 1984): 466-68.
       https://doi.org/10.1016/0017-9310(84)90295-3.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    if not buoyancy:
        return 1.0
    Ra = Gr*Pr
    if Ra < Rac:
        return 1.0

    k1 = 1.44/(1.0 + 0.018/Pr + 0.00136/(Pr*Pr))
    k2 = 75*exp(1.5*Pr**-0.5)

    t1 = (1.0 - Rac/Ra)
    t2 = k1 + 2.0*(Ra**(1.0/3.0)/k2)**(1.0 - log(Ra**(1.0/3.0)/k2))
    t3 = (Ra/5803.0)**(1.0/3.0) - 1.0

    if Rac != 1708:
        t4 = max(0.0, (Ra/Rac)**(1.0/3.0) - 1.0)
        t5 = (1.0 - exp(-0.95*t4))
    else:
        t5 = 1.0

    Nu = 1.0 + max(0.0, t1)*max(0.0, t2) + max(0.0, t3)*t5
    return Nu


def Nu_Nusselt_vertical_Thess(Pr, Gr, H=None, L=None):
    r'''Calculates the Nusselt number for natural convection between two
    theoretical vertical flat plates using the correlation by Thess [1]
    in [1]_. This is a variant on the horizontal Rayleigh-Benard classic heat
    transfer problem.
    This correlation supports different aspect ratios,
    so the plates can be real, finite objects and have their heat transfer
    accurately modeled. The recommended range of the correlation is H/L < 80.

    For 1e4 < Ra < 1e7:

    .. math::
        \text{Nu} = 0.42{Pr}^{0.012} {Ra}^{0.25} \left(\frac{H}{L}\right)^{-0.25}

    For 1e7 < Ra > 1e9 (or when geometry is unknown):

    .. math::
         \text{Nu} = 0.049{Ra}^{0.33}

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - plate
        temperature difference [-]
    H : float, optional
        Height of vertical plate, [m]
    L : float, optional
        Length of vertical plate, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to distance between the two plates, [-]

    Examples
    --------
    >>> Nu_Nusselt_vertical_Thess(.7, 3.21e6)
    6.112587569602785

    >>> Nu_Nusselt_vertical_Thess(.7, 3.21e6, L=10, H=1)
    28.79328626041646

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    Ra = Gr*Pr
    if Ra < 1e7 and H is not None and L is not None:
        return 0.42*Pr**0.012*Ra**0.25*(L/H)**0.25
    return 0.049*Ra**0.33



ratios_uninsulated_Catton = [0.125, 0.25, 0.5, 1, 2, 3, 4, 5, 6]
Racs_uninstulated_Catton = [[9802960, 1554480, 606001, 469377, 444995, 444363, 457007, 473725, 494741],
[1554480, 638754, 115596, 64270.8, 53529.7, 50816.4, 50136.1, 50088.7, 50410.1],
[606001, 115596, 48178.9, 14615.3, 11374.5, 9831.6, 9312, 9099.4, 8980.2],
[469377, 64270.8, 14615.3, 6974, 5138.2, 3906, 3633.6, 3446.2, 3358],
[444995, 53529.7, 11374.5, 5137.9, 3773.6, 2753.6, 2530.5, 2359.5, 2285.7],
[444363, 50816.4, 9831.6, 3906, 2753, 2557.4, 2337.2, 2174.44, 2101],
[457007, 50136.1, 9311.9, 3633.6, 2530.5, 2337.2, 2270.2, 2110.9, 2037.2],
[473725, 50088.6, 9099.4, 3446.2, 2359.5, 2174.4, 2110.9, 2081.7, 2007.8],
[494742, 50410.1, 8980.2, 3357.9, 2285.7, 2100.9, 2037.2, 2007.8, 1991.9]]

tck_uninstulated_Catton = implementation_optimize_tck([[0.125, 0.125, 0.125, 0.125, 0.41375910864088195,
                              0.5819413331927507, 1.9885569998423345, 2.8009586482973834,
                              3.922852887459219, 6.0, 6.0, 6.0, 6.0],
                             [0.125, 0.125, 0.125, 0.125, 0.4180739258304788,
                              0.6521218159098487, 1.4270223336187269,
                              2.89426640315332, 3.9239774081390215,
                              6.0, 6.0, 6.0, 6.0],
 [16.098194938851986, 14.026983058722742, 13.35866942808268, 13.043296359953983,
  13.008470795621905, 12.991279831677808, 13.040841344665466, 13.07803101947673,
  13.111789672293794, 14.074352449019207, 14.878522936155216, 11.151352953023258,
  11.096394321545977, 10.813773781060574, 10.796217122120712, 10.78189560829848,
  10.774336865714089, 10.78004622910552, 13.400086198278455, 11.369928815173187,
  11.82067779495709, 9.6860949637944, 9.686120336218499, 9.50952376562826,
  9.444619552074945, 9.452058024482865, 9.441608909473647, 12.933722760010111,
  10.873615956186896, 8.971126166473885, 8.520162104980807, 8.317346176887659,
  7.837750498437191, 7.78951404473208, 7.690715685713949, 7.695209247397283,
  13.025815591825872, 10.75723159025179, 9.734653433466208, 8.569056561731081,
  8.77031704228521, 7.853798846698488, 7.939088236475908, 7.748880239519593,
  7.785611785518214, 12.992898431724237, 10.728320934519346, 9.37520794405935,
  8.247995842200584, 7.753730020752022, 7.937553314495094, 7.6598493250444255,
  7.673199977054488, 7.63790748099515, 13.041869920313422, 10.713059500923494,
  9.364505568407685, 8.18000143764639, 7.927764179244221, 7.660718938605501,
  7.85174473958641, 7.5354388646400965, 7.614740168201775, 13.077057211283323,
  10.706667262420716, 9.341451094646674, 8.122270764822368, 7.671593316397699,
  7.697000470802994, 7.530680469875164, 7.720180133976149, 7.59900173760075,
  13.111791693551362, 10.711047679739433, 9.339770955175847, 8.117021359757253,
  7.727537757463738, 7.654072928976537, 7.607359118625173, 7.602197791148399,
  7.596844236081228], 3, 3])

ratios_insulated_Catton = [0.125, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 12]
Racs_instulated_Catton = [[3011718, 333013, 70040, 37689, 39798, 36262, 37058, 35875, 36209, 35664, 35794, 35486, 35556, 35380, 35451, 35193],
[333013, 203163, 28452, 11962, 12540, 11020, 11251, 10757, 10858, 10635, 10666, 10544, 10571, 10499, 10518, 10426],
[70040, 28452, 17307, 5262, 5341, 4524, 4567, 4330, 4355, 4245, 4261, 4186, 4196, 4158, 4165, 4118],
[37689, 11962, 5262, 3446, 3270, 2789, 2754, 2622, 2609, 2552, 2545, 2502, 2498, 2480, 2447, 2453],
[39798, 12540, 5341, 3270, None, None, None, None, None, None, None, None, None, None, None, None],
[36262, 11020, 4524, 2789, None, 2276, 2222, 2121, 2098, 2057, 2044, 2009, 2001, 1989, 1984, 1967],
[37058, 11251, 4567, 2754, None, 2222, None, None, None, None, None, None, None, None, None, None],
[35875, 10757, 4330, 2622, None, 2121, None, 2004, 1978, 1941, 1927, 1897, 1888, 1879, 1871, 1855],
[36209, 10858, 4355, 2609, None, 2098, None, 1978, None, None, None, None, None, None, None, None],
[35664, 10635, 4245, 2552, None, 2057, None, 1941, None, 1894, 1878, 1852, 1842, 1833, 1826, 1808],
[35794, 10666, 4261, 2545, None, 2044, None, 1927, None, 1878, None, None, None, None, None, None],
[35486, 10544, 4186, 2502, None, 2009, None, 1897, None, 1852, None, None, None, 1810, 1803, 1783],
[35556, 10571, 4196, 2498, None, 2001, None, 1888, None, 1842, None, None, None, None, None, None],
[35380, 10499, 4158, 2480, None, 1989, None, 1879, None, 1833, None, 1810, None, 1797, 1789, 1768],
[35451, 10518, 4165, 2447, None, 1984, None, 1871, None, 1826, None, 1803, None, 1789, None, None],
[35193, 10426, 4118, 2453, None, 1967, None, 1855, None, 1808, None, 1783, None, 1768, None, 1741]]


tck_insulated_Catton = implementation_optimize_tck([[0.125, 0.125, 0.2165763979498294, 0.25, 0.4948545767149843,
                                                     0.8432690088415454, 2.297018168305444, 5.324310151069744, 12.0, 12.0],
 [0.125, 0.125, 0.125, 0.37135574365684176, 0.8160817162671293, 1.1103105500488575,
  1.9000136398530074, 3.521092600950009, 12.0, 12.0, 12.0],
  [14.917942380813974, 12.196391449028951, 10.665084931671647, 10.531834082947338,
   10.57637568816619, 10.486173564722383, 10.471864979770599, 10.468190753935556,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.715841947316376, 12.462417612931137,
   9.174421085152083, 9.411191211042704, 9.409695481542864, 9.28122664900159,
   9.249608368005552, 9.251639244971427, 11.165512470689693, 10.01308504970903,
   9.75292707527754, 8.509349912597454, 8.566854764542974, 8.372517445356857,
   8.32618713246236, 8.329704835832104, 10.56848779064929, 9.163970117017675,
   8.369187019066972, 8.19799054440329, 8.087508877612247, 7.896372367041187,
   7.806891615973793, 7.835687464634469, 10.509836235182163, 9.041210689705586,
   8.118960504225761, 7.909354018896528, 7.735269232380504, 7.614379036546508,
   7.4775491512154515, 7.529024952770015, 10.474423221467699, 8.98482837851057,
   8.036532362247245, 7.822308882170893, 7.6362269726600065, 7.539826337638537,
   7.459554042916101, 7.480930154132415, 10.469149134470264, 8.978694786931275,
   8.024134988827441, 7.811393154091167, 7.627457342156321, 7.521833838146938,
   7.4376750879045455, 7.462202956737165], 1, 2])


def Rac_Nusselt_Rayleigh(H, L, W, insulated=True):
    r'''Calculates the critical Rayleigh number for free convection to begin
    in the Nusselt-Rayleigh parallel horizontal plate scenario. There are
    actually two cases - one for the top plate to be insulated (adiabatic) and
    the other where it has infinite thermal conductivity/is infinitely thin or
    not present (perfectly conducting). All real cases will lie between the
    two.

    Parameters
    ----------
    H : float
        Distance between the two plates, [m]
    L : float
        Length of the plates, [m]
    W : float
        Width of the plates, [m]
    insulated : bool, optional
        Whether the top plate is insulated or uninsulated, [-]

    Returns
    -------
    Rac : float
        Critical Rayleigh number, [-]

    Examples
    --------
    >>> Rac_Nusselt_Rayleigh(1, .5, 2, False)
    2530.500000000005
    >>> Rac_Nusselt_Rayleigh(1, .5, 2, True)
    2071.0089443385655

    Notes
    -----
    Splines have been fit to data in [1]_ for the uninsulated case and [2]_
    for the insulated case. The data is presented in the original papers and
    in [3]_.

    References
    ----------
    .. [1] Catton, Ivan. "Effect of Wall Conduction on the Stability of a Fluid
       in a Rectangular Region Heated from Below." Journal of Heat Transfer 94,
       no. 4 (November 1, 1972): 446-52. https://doi.org/10.1115/1.3449966.
    .. [2] Catton, Ivan. "Convection in a Closed Rectangular Region: The Onset
       of Motion." Journal of Heat Transfer 92, no. 1 (February 1, 1970):
       186-88. https://doi.org/10.1115/1.3449626.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    H_L_ratio = min(max(H/L, 0.125), 12.0)
    W_L_ratio = min(max(W/L, 0.125), 12.0)

    if insulated:
        Rac = exp(bisplev(W_L_ratio, H_L_ratio, tck_insulated_Catton))
    else:
        Rac = exp(bisplev(W_L_ratio, H_L_ratio, tck_uninstulated_Catton))
    return Rac


uninsulated_disk_coeffs = [1.3624571738082523, -0.24301326192178863, -6.152310426160362,
                           1.1950540229805053, 11.401090141352329, -2.405543860763877,
                           -11.091871509655324, 2.519761389270987, 5.992609902331248,
                           -1.4345227368881952, -1.7445130176764998, 0.42892571421446996,
                           0.22897205478499438, -0.042179780698649895, -0.01904413256783342,
                           0.006771075600246057, 0.13171026423861615]


insulated_disk_coeffs = [0.2173851248644496, 0.09672312658254612, -1.0800494968302843,
                         -0.3323452633903514, 2.1789014174652115, 0.43391756058946473,
                         -2.275756526433769, -0.29309565826688255, 1.3153930583762103,
                         0.14707146242791974, -0.44891166228441826, -0.045070571352735386,
                         0.08693822836596571, 0.010343944709216, -0.01325209778273359,
                         0.0035707992137628142, 0.13258956599554672]


def Rac_Nusselt_Rayleigh_disk(H, D, insulated=True):
    r'''Calculates the critical Rayleigh number for free convection to begin
    in the parallel horizontal disk scenario. There are
    actually two cases - one for the top plate to be insulated (adiabatic) and
    the other where it has infinite thermal conductivity/is infinitely thin or
    not present (perfectly conducting). All real cases will lie between the
    two.

    Parameters
    ----------
    H : float
        Distance between the two disks, [m]
    D : float
        Diameter of the two disks, [m]
    insulated : bool, optional
        Whether the top plate is insulated or uninsulated, [-]

    Returns
    -------
    Rac : float
        Critical Rayleigh number, [-]

    Examples
    --------
    >>> Rac_Nusselt_Rayleigh_disk(H=1, D=.4, insulated=False)
    151199.9999999945

    >>> Rac_Nusselt_Rayleigh_disk(H=1, D=4, insulated=False)
    1891.520931853363

    >>> Rac_Nusselt_Rayleigh_disk(2, 1, True)
    24347.31479211917

    Notes
    -----
    The range of data covered by this function is `D`/`H` from 0.4 to infinity.
    As inifinity is not well suited to polynomial form, the upper limit is
    6 in actuality. Values outside that range are rounded to the limits.

    This function provides 17-coefficient polynomial fits to interpolate in the
    table of values in [1]_. The source of the coefficients is cited as being
    from [2]_.

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Buell, J. C., and I. Catton. "The Effect of Wall Conduction on the
       Stability of a Fluid in a Right Circular Cylinder Heated From Below."
       Journal of Heat Transfer 105, no. 2 (May 1, 1983): 255-60.
       https://doi.org/10.1115/1.3245571.
    '''
    x = min(max(D/H, 0.4), 6.0)
    if insulated:
        coeffs = insulated_disk_coeffs
    else:
        coeffs = uninsulated_disk_coeffs
    return exp(1.0/horner(coeffs, 0.357142857142857151*(x - 3.2)))


### Free convection vertical helical coil

def Nu_vertical_helical_coil_Ali(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    helical coil inside a tank or other vessel according to the Ali [1]_
    correlation.

    .. math::
        Nu_L = 0.555Gr_L^{0.301} Pr^{0.314}

    Parameters
    ----------
    Pr : float
        Prandtl number of the fluid surrounding the coil with properties
        evaluated at bulk conditions or as described in the notes [-]
    Gr : float
        Prandtl number of the fluid surrounding the coil with properties
        evaluated at bulk conditions or as described in the notes
        (for the two temperatures, use the average coil fluid temperature and
        the temperature of the fluid outside the coil) [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to the total length of the helical coil
        (and bulk thermal conductivity), [-]

    Notes
    -----
    In [1]_, the temperature at which the fluid surrounding the coil's
    properties were evaluated at was calculated in an unusual fashion. The
    average temperature of the fluid inside the coil
    :math:`(T_{in} + T_{out})/2` is averaged with the fluid outside the coil's
    temperature.

    The correlation is valid for Prandtl numbers between 4.4 and 345,
    and tank diameter/coil outer diameter ratios between 10 and 30.

    Examples
    --------
    >>> Nu_vertical_helical_coil_Ali(4.4, 1E11)
    1808.57749972

    References
    ----------
    .. [1] Ali, Mohamed E. "Natural Convection Heat Transfer from Vertical
       Helical Coils in Oil." Heat Transfer Engineering 27, no. 3 (April 1,
       2006): 79-85.
    '''
    return 0.555*Gr**0.301*Pr**0.314


def Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    helical coil inside a tank or other vessel according to the Prabhanjan,
    Rennie, and Raghavan [1]_ correlation.

    .. math::
        Nu_H = 0.0749\text{Ra}_H^{0.3421}

    The range of Rayleigh numbers is as follows:

    .. math::
        9 \times 10^{9} < \text{Ra} < 4 \times 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number calculated with the film temperature -
        wall and temperature very far from the coil average, [-]
    Gr : float
        Grashof number calculated with the film temperature -
        wall and temperature very far from the coil average,
        and using the total height of the coil [-]

    Returns
    -------
    Nu : float
        Nusselt number using the total height of the coil
        and the film temperature, [-]

    Notes
    -----
    [1]_ also has several other equations using different characteristic
    lengths.

    Examples
    --------
    >>> Nu_vertical_helical_coil_Prabhanjan_Rennie_Raghavan(4.4, 1E11)
    720.6211067718227

    References
    ----------
    .. [1] Prabhanjan, Devanahalli G., Timothy J. Rennie, and G. S. Vijaya
       Raghavan. "Natural Convection Heat Transfer from Helical Coiled Tubes."
       International Journal of Thermal Sciences 43, no. 4 (April 1, 2004):
       359-65.
    '''
    Ra = Pr*Gr
    return 0.0749*Ra**0.3421


def Q_enclosure_natural(k_eff, A, T1, T2, L):
    r'''Calculates the steady-state heat transfer rate through a planar
    enclosure (such as a double-pane window) by natural convection,
    according to Eq. 9-40 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \dot{Q} = k_{eff} A \frac{T_1 - T_2}{L}

    Parameters
    ----------
    k_eff : float
        Effective thermal conductivity of the enclosed fluid, [W/(m*K)]
    A : float
        Heat transfer surface area of one of the plates, [m^2]
    T1 : float
        Temperature of the hotter surface, [K] or [deg C]
    T2 : float
        Temperature of the colder surface, [K] or [deg C]
    L : float
        Distance between the two plates (gap thickness), [m]

    Returns
    -------
    Q : float
        Rate of heat transfer through the enclosure, [W]

    Examples
    --------
    Example 9-4 from [1]_:
    >>> Q_enclosure_natural(0.03385, 1.6, 12.0, 2.0, 0.02) # doctest: +ELLIPSIS
    27.08...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-40, p. 553.
    '''
    if L <= 0:
        raise ValueError("Gap thickness L must be positive.")
    return k_eff * A * (T1 - T2) / L


def k_eff_enclosure(k, Nu):
    r'''Calculates the effective thermal conductivity of an enclosure fluid,
    according to Eq. 9-41 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        k_{eff} = k \cdot Nu

    Parameters
    ----------
    k : float
        Actual thermal conductivity of the fluid, [W/(m*K)]
    Nu : float
        Nusselt number of the enclosure flow [-]

    Returns
    -------
    k_eff : float
        Effective thermal conductivity (>= k), [W/(m*K)]

    Notes
    -----
    When Nu <= 1, convection currents are negligible and heat transfer
    is purely conductive, so k_eff = k.

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-41, p. 553.
    '''
    Nu_eff = max(1.0, Nu)
    return k * Nu_eff


def Nu_horizontal_enclosure_Jakob(Ra, regime='auto'):
    r'''Calculates the Nusselt number for natural convection inside a horizontal
    enclosure containing air (or gases with 0.5 < Pr < 2) heated from below,
    according to Jakob (1949) and Eqs. 9-44, 9-45 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.195 Ra_L^{1/4} \quad (10^4 < Ra_L < 4 \times 10^5) \quad \text{[Eq. 9-44]} \\
        Nu = 0.068 Ra_L^{1/3} \quad (4 \times 10^5 \le Ra_L < 10^7) \quad \text{[Eq. 9-45]}

    For :math:`Ra_L < 1708`, :math:`Nu = 1` (pure conduction).

    Parameters
    ----------
    Ra : float
        Rayleigh number based on plate spacing L [-]
    regime : str, optional
        'auto', 'laminar', or 'turbulent' [-]

    Returns
    -------
    Nu : float
        Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eqs. 9-44, 9-45, p. 554.
    .. [2] Jakob, M. Heat Transfer. Vol. 1. New York: John Wiley & Sons, 1949.
    '''
    if Ra < 1708:
        return 1.0
    r = regime.lower()
    if r == 'laminar':
        if Ra < 1e4 or Ra > 4e5:
            warnings.warn("Ra={:.3e} is outside recommended range 10^4 < Ra < 4e5 for Jakob laminar correlation.".format(Ra), RuntimeWarning)
        return 0.195 * Ra**0.25
    elif r == 'turbulent':
        if Ra < 4e5 or Ra > 1e7:
            warnings.warn("Ra={:.3e} is outside recommended range 4e5 < Ra < 10^7 for Jakob turbulent correlation.".format(Ra), RuntimeWarning)
        return 0.068 * Ra**(1.0/3.0)
    elif r == 'auto':
        if Ra < 4e5:
            return 0.195 * Ra**0.25
        else:
            return 0.068 * Ra**(1.0/3.0)
    else:
        raise ValueError("Regime '{}' not recognized. Use 'auto', 'laminar', or 'turbulent'.".format(regime))


def Nu_horizontal_enclosure_Globe_Dropkin(Pr, Ra):
    r'''Calculates the Nusselt number for natural convection inside a horizontal
    enclosure heated from below containing liquids (e.g., water, silicone oil,
    mercury), according to Globe & Dropkin (1959) and Eq. 9-46 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.069 Ra_L^{1/3} Pr^{0.074} \quad (3 \times 10^5 < Ra_L < 7 \times 10^9)

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on plate spacing L [-]

    Returns
    -------
    Nu : float
        Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-46, p. 554.
    .. [2] Globe, S., and D. Dropkin. "Natural-Convection Heat Transfer in
       Liquids Confined by Two Horizontal Plates and Heated from Below."
       Journal of Heat Transfer 81 (1959): 24-28.
    '''
    if Ra < 3e5 or Ra > 7e9:
        warnings.warn("Ra={:.3e} is outside recommended range 3e5 < Ra < 7e9 for Globe-Dropkin correlation.".format(Ra), RuntimeWarning)
    return 0.069 * (Ra**(1.0/3.0)) * (Pr**0.074)


def Nu_horizontal_enclosure_Hollands(Ra):
    r'''Calculates the Nusselt number for natural convection inside a horizontal
    enclosure heated from below, according to Hollands et al. (1976) and
    Eq. 9-47 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 1 + 1.44 \left[ 1 - \frac{1708}{Ra_L} \right]^+ + \left[ \frac{Ra_L^{1/3}}{18} - 1 \right]^+ \quad (Ra_L < 10^8)

    where :math:`[\cdot]^+` sets negative values to zero.

    Parameters
    ----------
    Ra : float
        Rayleigh number based on plate spacing L [-]

    Returns
    -------
    Nu : float
        Nusselt number based on spacing L, [-]

    Notes
    -----
    Recommended for air and data correlates well for moderate Prandtl number
    liquids (such as water) for :math:`Ra_L < 10^5`.

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-47, p. 554.
    .. [2] Hollands, K. G. T., T. E. Unny, G. D. Raithby, and L. Konicek.
       "Free Convective Heat Transfer Across Inclined Air Layers." Journal of
       Heat Transfer 98 (1976): 189-193.
    '''
    if Ra <= 0:
        return 1.0
    term1 = 1.44 * max(0.0, 1.0 - 1708.0 / Ra)
    term2 = max(0.0, (Ra**(1.0/3.0)) / 18.0 - 1.0)
    return 1.0 + term1 + term2


def Nu_inclined_enclosure_Hollands(Ra, theta, H, L):
    r'''Calculates the Nusselt number for natural convection inside an inclined
    rectangular enclosure (e.g. flat-plate solar collector) with large aspect
    ratio :math:`H/L \ge 12`, according to Hollands et al. (1976) and
    Eq. 9-48 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 1 + 1.44 \left[ 1 - \frac{1708}{Ra_L \cos\theta} \right]^+
        \left[ 1 - \frac{1708 (\sin 1.8\theta)^{1.6}}{Ra_L \cos\theta} \right]
        + \left[ \frac{(Ra_L \cos\theta)^{1/3}}{18} - 1 \right]^+

    for :math:`Ra_L \le 10^5`, :math:`0^\circ \le \theta \le 70^\circ`, and :math:`H/L \ge 12`.
    Here :math:`\theta` is the tilt angle measured from the horizontal.

    Parameters
    ----------
    Ra : float
        Rayleigh number based on plate spacing L [-]
    theta : float
        Tilt angle from the horizontal in degrees [deg] (0 <= theta <= 70)
    H : float
        Height (length along incline) of the enclosure, [m]
    L : float
        Spacing between the two parallel plates, [m]

    Returns
    -------
    Nu : float
        Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-48, p. 554.
    '''
    aspect = H / L
    if aspect < 12.0 - 1e-9:
        warnings.warn("Aspect ratio H/L={:.2f} is less than recommended minimum 12 for Hollands inclined correlation.".format(aspect), RuntimeWarning)
    if theta < 0.0 or theta > 70.0:
        warnings.warn("Tilt angle theta={:.1f} deg is outside recommended range [0, 70] deg.".format(theta), RuntimeWarning)

    rad_theta = radians(theta)
    cos_theta = cos(rad_theta)
    Ra_cos = Ra * cos_theta

    if Ra_cos <= 0:
        return 1.0

    bracket1 = max(0.0, 1.0 - 1708.0 / Ra_cos)
    sin_term = sin(1.8 * rad_theta)
    factor2 = 1.0 - 1708.0 * (max(0.0, sin_term)**1.6) / Ra_cos
    bracket3 = max(0.0, (Ra_cos**(1.0/3.0)) / 18.0 - 1.0)

    return 1.0 + 1.44 * bracket1 * factor2 + bracket3


def critical_angle_inclined_enclosure(aspect_ratio):
    r'''Determines the critical angle theta_cr (in degrees) for an inclined
    rectangular enclosure as a function of aspect ratio H/L, based on
    Table 9-2 in Çengel & Ghajar (5th Ed) [1]_.

    Parameters
    ----------
    aspect_ratio : float
        Aspect ratio H/L of the enclosure [-]

    Returns
    -------
    theta_cr : float
        Critical tilt angle from horizontal, [deg]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Table 9-2, p. 554.
    .. [2] Catton, I. "Natural Convection in Enclosures." In Proc. 6th Int. Heat
       Transfer Conf., Vol. 6, pp. 13-31, 1978.
    '''
    pts = [(1.0, 25.0), (3.0, 53.0), (6.0, 60.0), (12.0, 67.0)]
    if aspect_ratio <= 1.0:
        return 25.0
    elif aspect_ratio >= 12.0:
        return 70.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i+1]
        if x0 <= aspect_ratio <= x1:
            return y0 + (aspect_ratio - x0) * (y1 - y0) / (x1 - x0)
    return 70.0


def Nu_inclined_enclosure_Catton(Ra, theta, aspect_ratio, Pr=0.71):
    r'''Calculates the Nusselt number for natural convection inside an inclined
    rectangular enclosure with small aspect ratio :math:`H/L < 12`, according to
    Catton (1978), Ayyaswamy & Catton (1973), and Eqs. 9-49, 9-50 in
    Çengel & Ghajar (5th Ed) [1]_.

    - For :math:`0^\circ < \theta < \theta_{cr}`:
      .. math::
          Nu = Nu_0 \left( \frac{Nu_{90}}{Nu_0} \right)^{\theta / \theta_{cr}} (\sin\theta_{cr})^{\theta / (4 \theta_{cr})} \quad \text{[Eq. 9-49]}

    - For :math:`\theta_{cr} \le \theta \le 90^\circ`:
      .. math::
          Nu = Nu_{90} (\sin\theta)^{1/4} \quad \text{[Eq. 9-50]}

    Parameters
    ----------
    Ra : float
        Rayleigh number based on plate spacing L [-]
    theta : float
        Tilt angle from horizontal in degrees [deg] (0 <= theta <= 90)
    aspect_ratio : float
        Aspect ratio H/L of the enclosure [-]
    Pr : float, optional
        Prandtl number of fluid. Default is 0.71 (air) [-]

    Returns
    -------
    Nu : float
        Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eqs. 9-49, 9-50, Table 9-2, p. 554.
    '''
    theta_cr = critical_angle_inclined_enclosure(aspect_ratio)
    Nu_0 = Nu_horizontal_enclosure_Hollands(Ra)
    Nu_90 = Nu_vertical_enclosure_Cengel(Pr, Ra, aspect_ratio)

    if theta <= 0.0:
        return Nu_0
    elif theta >= 90.0:
        return Nu_90

    rad_theta = radians(theta)
    rad_theta_cr = radians(theta_cr)

    if theta < theta_cr:
        ratio = Nu_90 / max(1e-12, Nu_0)
        exp1 = theta / theta_cr
        exp2 = theta / (4.0 * theta_cr)
        return Nu_0 * (ratio**exp1) * (sin(rad_theta_cr)**exp2)
    else:
        return Nu_90 * (sin(rad_theta)**0.25)


def Nu_inclined_enclosure_Arnold(Nu_vertical, theta):
    r'''Calculates the Nusselt number for natural convection inside an enclosure
    tilted more than 90 degrees from the horizontal, according to Arnold et al. (1974)
    and Eq. 9-51 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 1 + (Nu_{\theta = 90^\circ} - 1) \sin\theta \quad (90^\circ < \theta < 180^\circ)

    Parameters
    ----------
    Nu_vertical : float
        Nusselt number for the corresponding vertical enclosure (theta = 90 deg) [-]
    theta : float
        Tilt angle from horizontal in degrees [deg] (90 < theta < 180)

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-51, p. 555.
    '''
    rad_theta = radians(theta)
    return 1.0 + (Nu_vertical - 1.0) * sin(rad_theta)


def Nu_vertical_enclosure_Berkovsky_Polevikov_1(Pr, Ra):
    r'''Calculates Nusselt number for natural convection inside a vertical
    rectangular enclosure with aspect ratio :math:`1 < H/L < 2`, according to
    Berkovsky & Polevikov (1977) and Eq. 9-52 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.18 \left( \frac{Pr}{0.2 + Pr} Ra_L \right)^{0.29}

    Applicable for any Prandtl number when :math:`Ra_L Pr / (0.2 + Pr) > 10^3`.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on enclosure width L [-]

    Returns
    -------
    Nu : float
        Average Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-52, p. 555.
    '''
    param = (Pr / (0.2 + Pr)) * Ra
    if param < 1e3:
        warnings.warn("Ra*Pr/(0.2+Pr)={:.2e} is less than recommended minimum 10^3 for Berkovsky-Polevikov correlation 1.".format(param), RuntimeWarning)
    if param <= 0:
        return 1.0
    return 0.18 * (param**0.29)


def Nu_vertical_enclosure_Berkovsky_Polevikov_2(Pr, Ra, aspect_ratio):
    r'''Calculates Nusselt number for natural convection inside a vertical
    rectangular enclosure with aspect ratio :math:`2 < H/L < 10`, according to
    Berkovsky & Polevikov (1977) and Eq. 9-53 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.22 \left( \frac{Pr}{0.2 + Pr} Ra_L \right)^{0.28} \left( \frac{H}{L} \right)^{-1/4}

    Applicable for any Prandtl number and :math:`Ra_L < 10^{10}`.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on enclosure width L [-]
    aspect_ratio : float
        Aspect ratio H/L (2 < H/L < 10) [-]

    Returns
    -------
    Nu : float
        Average Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-53, p. 555.
    '''
    if Ra > 1e10:
        warnings.warn("Ra={:.2e} exceeds recommended maximum 10^10 for Berkovsky-Polevikov correlation 2.".format(Ra), RuntimeWarning)
    param = (Pr / (0.2 + Pr)) * Ra
    if param <= 0 or aspect_ratio <= 0:
        return 1.0
    return 0.22 * (param**0.28) * (aspect_ratio**(-0.25))


def Nu_vertical_enclosure_MacGregor_Emery_laminar(Pr, Ra, aspect_ratio):
    r'''Calculates Nusselt number for laminar natural convection inside a vertical
    rectangular enclosure with aspect ratio :math:`10 < H/L < 40`, according to
    MacGregor & Emery (1969) and Eq. 9-54 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.42 Ra_L^{1/4} Pr^{0.012} \left( \frac{H}{L} \right)^{-0.3}

    Applicable for :math:`10 < H/L < 40`, :math:`1 < Pr < 2 \times 10^4`, and :math:`10^4 < Ra_L < 10^7`.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on enclosure width L [-]
    aspect_ratio : float
        Aspect ratio H/L (10 < H/L < 40) [-]

    Returns
    -------
    Nu : float
        Average Nusselt number based on spacing L, [-]

    Examples
    --------
    Example 9-4 from [1]_:
    >>> Nu_vertical_enclosure_MacGregor_Emery_laminar(0.7344, 1.050e4, 40.0) # doctest: +ELLIPSIS
    1.401...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-54, p. 555.
    '''
    if Ra <= 0 or aspect_ratio <= 0:
        return 1.0
    return 0.42 * (Ra**0.25) * (Pr**0.012) * (aspect_ratio**(-0.3))


def Nu_vertical_enclosure_MacGregor_Emery_turbulent(Ra):
    r'''Calculates Nusselt number for turbulent natural convection inside a vertical
    rectangular enclosure with aspect ratio :math:`1 < H/L < 40`, according to
    MacGregor & Emery (1969) and Eq. 9-55 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.046 Ra_L^{1/3}

    Applicable for :math:`1 < H/L < 40`, :math:`1 < Pr < 20`, and :math:`10^6 < Ra_L < 10^9`.

    Parameters
    ----------
    Ra : float
        Rayleigh number based on enclosure width L [-]

    Returns
    -------
    Nu : float
        Average Nusselt number based on spacing L, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-55, p. 555.
    '''
    if Ra <= 0:
        return 1.0
    return 0.046 * (Ra**(1.0/3.0))


def Nu_vertical_enclosure_Cengel(Pr, Ra, aspect_ratio):
    r'''Master selector for natural convection inside a vertical rectangular
    enclosure, selecting between Berkovsky-Polevikov and MacGregor-Emery
    correlations based on aspect ratio H/L and Rayleigh number, according to
    Section 9-5 in Çengel & Ghajar (5th Ed) [1]_.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on enclosure width L [-]
    aspect_ratio : float
        Aspect ratio H/L of the enclosure [-]

    Returns
    -------
    Nu : float
        Average Nusselt number, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Section 9-5, p. 555.
    '''
    if aspect_ratio <= 2.0:
        return Nu_vertical_enclosure_Berkovsky_Polevikov_1(Pr, Ra)
    elif aspect_ratio <= 10.0:
        return Nu_vertical_enclosure_Berkovsky_Polevikov_2(Pr, Ra, aspect_ratio)
    else:
        if Ra < 1e6:
            return Nu_vertical_enclosure_MacGregor_Emery_laminar(Pr, Ra, aspect_ratio)
        else:
            return Nu_vertical_enclosure_MacGregor_Emery_turbulent(Ra)


def F_concentric_cylinders(Di, Do):
    r'''Calculates geometric factor F_cyl for natural convection in the annular
    space between two horizontal concentric cylinders, according to
    Eq. 9-58 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        F_{cyl} = \frac{[\ln(D_o / D_i)]^4}{L_c^3 (D_i^{-3/5} + D_o^{-3/5})^5}

    where :math:`L_c = (D_o - D_i) / 2`.

    Parameters
    ----------
    Di : float
        Inner cylinder diameter, [m]
    Do : float
        Outer cylinder diameter, [m]

    Returns
    -------
    F_cyl : float
        Concentric cylinders geometric factor, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-58, p. 556.
    '''
    if Do <= Di or Di <= 0:
        raise ValueError("Outer diameter Do must be strictly greater than inner diameter Di > 0.")
    Lc = (Do - Di) / 2.0
    num = (log(Do / Di))**4
    denom = (Lc**3) * ((Di**(-0.6) + Do**(-0.6))**5)
    return num / denom


def k_eff_concentric_cylinders_Raithby_Hollands(Pr, Ra, Di, Do, k):
    r'''Calculates effective thermal conductivity for natural convection between
    horizontal isothermal concentric cylinders, according to Raithby & Hollands (1975)
    and Eq. 9-57 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \frac{k_{eff}}{k} = 0.386 \left( \frac{Pr}{0.861 + Pr} \right)^{1/4} (F_{cyl} Ra_L)^{1/4}

    where :math:`L_c = (D_o - D_i) / 2`.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on gap width Lc = (Do - Di)/2 [-]
    Di : float
        Inner cylinder diameter, [m]
    Do : float
        Outer cylinder diameter, [m]
    k : float
        Actual thermal conductivity of fluid, [W/(m*K)]

    Returns
    -------
    k_eff : float
        Effective thermal conductivity (>= k), [W/(m*K)]

    Notes
    -----
    Applicable for :math:`0.70 \le Pr \le 6000` and :math:`10^2 \le F_{cyl} Ra_L \le 10^7`.
    For :math:`F_{cyl} Ra_L < 100`, natural convection is negligible and :math:`k_{eff} = k`.

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-57, p. 555.
    '''
    Fcyl = F_concentric_cylinders(Di, Do)
    param = Fcyl * Ra
    if param < 100.0:
        return k
    ratio = 0.386 * ((Pr / (0.861 + Pr))**0.25) * (param**0.25)
    return max(k, k * ratio)


def Q_concentric_cylinders_natural(k_eff, Di, Do, Ti, To):
    r'''Calculates the steady natural convection heat transfer rate per unit
    length through the annular space between concentric horizontal cylinders,
    according to Eq. 9-56 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \dot{Q} = \frac{2 \pi k_{eff}}{\ln(D_o / D_i)} (T_i - T_o) \quad \text{[W/m]}

    Parameters
    ----------
    k_eff : float
        Effective thermal conductivity of the annular fluid, [W/(m*K)]
    Di : float
        Inner cylinder diameter, [m]
    Do : float
        Outer cylinder diameter, [m]
    Ti : float
        Inner cylinder surface temperature, [K] or [deg C]
    To : float
        Outer cylinder surface temperature, [K] or [deg C]

    Returns
    -------
    Q_per_m : float
        Heat transfer rate per unit length, [W/m]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-56, p. 555.
    '''
    if Do <= Di or Di <= 0:
        raise ValueError("Outer diameter Do must be strictly greater than inner diameter Di > 0.")
    return (2.0 * pi * k_eff / log(Do / Di)) * (Ti - To)


def F_concentric_spheres(Di, Do):
    r'''Calculates geometric factor F_sph for natural convection in the space
    between two concentric spheres, according to Eq. 9-61 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        F_{sph} = \frac{L_c}{(D_i D_o)^4 (D_i^{-7/5} + D_o^{-7/5})^5}

    where :math:`L_c = (D_o - D_i) / 2`.

    Parameters
    ----------
    Di : float
        Inner sphere diameter, [m]
    Do : float
        Outer sphere diameter, [m]

    Returns
    -------
    F_sph : float
        Concentric spheres geometric factor, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-61, p. 556.
    '''
    if Do <= Di or Di <= 0:
        raise ValueError("Outer diameter Do must be strictly greater than inner diameter Di > 0.")
    Lc = (Do - Di) / 2.0
    num = Lc
    denom = ((Di * Do)**4) * ((Di**(-1.4) + Do**(-1.4))**5)
    return num / denom


def k_eff_concentric_spheres_Raithby_Hollands(Pr, Ra, Di, Do, k):
    r'''Calculates effective thermal conductivity for natural convection between
    isothermal concentric spheres, according to Raithby & Hollands (1975)
    and Eq. 9-60 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \frac{k_{eff}}{k} = 0.74 \left( \frac{Pr}{0.861 + Pr} \right)^{1/4} (F_{sph} Ra_L)^{1/4}

    where :math:`L_c = (D_o - D_i) / 2`.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Ra : float
        Rayleigh number based on gap width Lc = (Do - Di)/2 [-]
    Di : float
        Inner sphere diameter, [m]
    Do : float
        Outer sphere diameter, [m]
    k : float
        Actual thermal conductivity of fluid, [W/(m*K)]

    Returns
    -------
    k_eff : float
        Effective thermal conductivity (>= k), [W/(m*K)]

    Notes
    -----
    Applicable for :math:`0.70 \le Pr \le 4200` and :math:`10^2 \le F_{sph} Ra_L \le 10^4`.
    If :math:`k_{eff} < k`, :math:`k_{eff} = k`.

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-60, p. 556.
    '''
    Fsph = F_concentric_spheres(Di, Do)
    param = Fsph * Ra
    ratio = 0.74 * ((Pr / (0.861 + Pr))**0.25) * (param**0.25)
    return max(k, k * ratio)


def Q_concentric_spheres_natural(k_eff, Di, Do, Ti, To):
    r'''Calculates steady natural convection heat transfer rate through the
    gap between concentric isothermal spheres, according to Eq. 9-59 in
    Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \dot{Q} = k_{eff} \frac{\pi D_i D_o}{L_c} (T_i - T_o) \quad \text{[W]}

    where :math:`L_c = (D_o - D_i) / 2`.

    Parameters
    ----------
    k_eff : float
        Effective thermal conductivity of fluid in the gap, [W/(m*K)]
    Di : float
        Inner sphere diameter, [m]
    Do : float
        Outer sphere diameter, [m]
    Ti : float
        Inner sphere surface temperature, [K] or [deg C]
    To : float
        Outer sphere surface temperature, [K] or [deg C]

    Returns
    -------
    Q : float
        Heat transfer rate, [W]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-59, p. 556.
    '''
    if Do <= Di or Di <= 0:
        raise ValueError("Outer diameter Do must be strictly greater than inner diameter Di > 0.")
    Lc = (Do - Di) / 2.0
    return k_eff * pi * Di * Do / Lc * (Ti - To)


def emissivity_effective_parallel_plates(eps1, eps2):
    r'''Calculates effective emissivity between two large parallel plates,
    according to Eq. 9-65 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \epsilon_{eff} = \frac{1}{1/\epsilon_1 + 1/\epsilon_2 - 1}

    Parameters
    ----------
    eps1 : float
        Emissivity of plate 1 (0 < eps1 <= 1) [-]
    eps2 : float
        Emissivity of plate 2 (0 < eps2 <= 1) [-]

    Returns
    -------
    eps_eff : float
        Effective emissivity between the plates, [-]

    Examples
    --------
    Ordinary glass surfaces (eps = 0.84):
    >>> emissivity_effective_parallel_plates(0.84, 0.84) # doctest: +ELLIPSIS
    0.7241...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-65, p. 557.
    '''
    if eps1 <= 0 or eps2 <= 0:
        raise ValueError("Emissivities must be positive.")
    return 1.0 / (1.0 / eps1 + 1.0 / eps2 - 1.0)
