'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import cos, log, radians, sin
import warnings

__all__ = ['Nu_vertical_plate_Churchill',
           'Nu_free_vertical_plate',
           'Nu_free_vertical_plate_methods',
           'Nu_horizontal_plate_McAdams',
           'Nu_horizontal_plate_VDI',
           'Nu_horizontal_plate_Rohsenow',
           'Nu_free_horizontal_plate',
           'Nu_free_horizontal_plate_methods',
           'Nu_sphere_Churchill',
           'Nu_vertical_cylinder_Griffiths_Davis_Morgan',
           'Nu_vertical_cylinder_Jakob_Linke_Morgan',
           'Nu_vertical_cylinder_Carne_Morgan',
           'Nu_vertical_cylinder_Eigenson_Morgan',
           'Nu_vertical_cylinder_Touloukian_Morgan',
           'Nu_vertical_cylinder_McAdams_Weiss_Saunders',
           'Nu_vertical_cylinder_Kreith_Eckert',
           'Nu_vertical_cylinder_Hanesian_Kalish_Morgan',
           'Nu_vertical_cylinder_Al_Arabi_Khamis',
           'Nu_vertical_cylinder_Popiel_Churchill',
           'Nu_vertical_cylinder',
           'Nu_vertical_cylinder_methods',
           'Nu_horizontal_cylinder_Churchill_Chu',
           'Nu_horizontal_cylinder_Kuehn_Goldstein',
           'Nu_horizontal_cylinder_Morgan',
           'Nu_horizontal_cylinder',
           'Nu_horizontal_cylinder_methods',
           'Nu_coil_Xin_Ebadian',
           'Nu_vertical_plate_laminar_Cengel',
           'Nu_vertical_plate_turbulent_Cengel',
           'Nu_vertical_plate_Cengel',
           'Nu_inclined_plate_Cengel',
           'Nu_horizontal_plate_Cengel',
           'is_vertical_cylinder_plate_like',
           'Nu_vertical_plate_isoflux_Cengel',
           'Ra_vertical_channel',
           'Nu_vertical_channel_isothermal_Bar_Cohen',
           'optimum_fin_spacing_isothermal',
           'Nu_optimum_fin_spacing',
           'Q_finned_surface_natural',
           'Ra_star_vertical_channel',
           'Nu_vertical_channel_isoflux_Bar_Cohen',
           'optimum_spacing_isoflux',
           'Q_PCBs_natural',
           'Nu_mixed_convection',
           'mixed_convection_regime',
           'h_air_natural_vertical_plate',
           'h_air_natural_horizontal_plate',
           'h_air_natural_horizontal_cylinder']


def Nu_vertical_plate_Churchill(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    plate according to the Churchill-Chu [1]_ correlation, also presented in
    [2]_. Plate must be isothermal; an alternate expression exists for constant
    heat flux.

    .. math::
        Nu_{L}=\left[0.825+\frac{0.387Ra_{L}^{1/6}}
        {[1+(0.492/Pr)^{9/16}]^{8/27}}\right]^2

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to height, [-]

    Notes
    -----
    Although transition from laminar to turbulent is discrete in reality, this
    equation provides a smooth transition in value from laminar to turbulent.
    Checked with the original source.

    Can be applied to vertical cylinders as well, subject to the criteria below:

    .. math::
        \frac{D}{L}\ge \frac{35}{Gr_L^{1/4}}

    Examples
    --------
    From [2]_, Example 9.2, matches:

    >>> Nu_vertical_plate_Churchill(0.69, 2.63E9)
    147.16185223770603

    References
    ----------
    .. [1] Churchill, Stuart W., and Humbert H. S. Chu. "Correlating Equations
       for Laminar and Turbulent Free Convection from a Vertical Plate."
       International Journal of Heat and Mass Transfer 18, no. 11
       (November 1, 1975): 1323-29. doi:10.1016/0017-9310(75)90243-4.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ra = Pr*Gr
    term = (0.825 + (0.387*Ra**(1/6.)*(1.0 + (Pr/0.492)**(-0.5625))**(-8.0/27.0)))
    return term*term

Nu_free_vertical_plate_all_methods = ["Churchill"]

def Nu_free_vertical_plate_methods(Pr, Gr, H=None, W=None, check_ranges=True):
    r'''This function returns a list of methods for calculating heat transfer
    coefficient for external free convection from a verical plate.

    Requires at a minimum a fluid's Prandtl number `Pr`, and the Grashof
    number `Gr` for the system fluid (which require T and P to obtain).

    `L` and `W` are not used by any correlations presently, but are included
    for future support.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    H : float, optional
        Height of vertical plate, [m]
    W : float, optional
        Width of the vertical plate, [m]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `Nu` with the given
        inputs, [-]

    Examples
    --------
    >>> Nu_free_vertical_plate_methods(0.69, 2.63E9)
    ['Churchill']
    '''
    return Nu_free_vertical_plate_all_methods

def Nu_free_vertical_plate(Pr, Gr, buoyancy=None, H=None, W=None, Method=None):
    r'''This function calculates the heat transfer coefficient for external
    free convection from a verical plate.

    Requires at a minimum a fluid's Prandtl number `Pr`, and the Grashof
    number `Gr` for the system fluid (which require T and P to obtain).

    `L` and `W` are not used by any correlations presently, but are included
    for future support.

    If no correlation's name is provided as `Method`, the 'Churchill'
    correlation is selected.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]
    H : float, optional
        Height of vertical plate, [m]
    W : float, optional
        Width of the vertical plate, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate height, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use;
        one of ('Churchill', ).

    Examples
    --------
    Turbulent example

    >>> Nu_free_vertical_plate(0.69, 2.63E9, False)
    147.16185223770603
    '''
    if Method is None:
        Method2 = 'Churchill'
    else:
        Method2 = Method
    if Method2 == 'Churchill':
        return Nu_vertical_plate_Churchill(Pr, Gr)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")


def Nu_horizontal_plate_McAdams(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection above a horizontal
    plate according to the McAdams [1]_ correlations. The plate must be
    isothermal. Four different equations are used, two each for laminar and
    turbulent; the two sets of correlations are required because if the plate
    is hot, buoyancy lifts the fluid off the plate and enhances free convection
    whereas if the plate is cold, the cold fluid above it settles on it and
    decreases the free convection.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to length, [-]

    Notes
    -----

    Examples
    --------
    >>> Nu_horizontal_plate_McAdams(5.54, 3.21e8, buoyancy=True)
    181.73121274384457
    >>> Nu_horizontal_plate_McAdams(5.54, 3.21e8, buoyancy=False)
    55.44564799362829

    >>> Nu_horizontal_plate_McAdams(.01, 3.21e8, buoyancy=True)
    22.857041558492334
    >>> Nu_horizontal_plate_McAdams(.01, 3.21e8, buoyancy=False)
    11.428520779246167

    References
    ----------
    .. [1] McAdams, William Henry. Heat Transmission. 3E. Malabar, Fla:
       Krieger Pub Co, 1985.
    '''
    Ra = Pr*Gr
    if buoyancy:
        if Ra <= 1E7:
            Nu = .54*Ra**0.25
        else:
            Nu = 0.15*Ra**(1.0/3.0)
    else:
        if Ra <= 1E10:
            Nu = .27*Ra**0.25
        else:
            Nu = .15*Ra**(1.0/3.0)
    return Nu


def Nu_horizontal_plate_VDI(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection above a horizontal
    plate according to the VDI [1]_ correlations. The plate must be
    isothermal. Three different equations are used, one each for laminar and
    turbulent for the heat transfer happening at upper surface case and one for
    the case of heat transfer happening at the lower surface. The lower surface
    correlation is recommened for the laminar flow regime.
    The two different sets of correlations are required because if the plate
    is hot, buoyancy lifts the fluid off the plate and enhances free convection
    whereas if the plate is cold, the cold fluid above it settles on it and
    decreases the free convection.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to length, [-]

    Notes
    -----
    The characteristic length suggested for use is as follows, with `a` and
    `b` being the length and width of the plate.

    .. math::
        L = \frac{ab}{2(a+b)}

    The buoyancy enhanced cases are from [2]_; the other is said to be from
    [3]_, although the equations there not quite the same and do not include
    the Prandtl number correction.

    Examples
    --------
    >>> Nu_horizontal_plate_VDI(5.54, 3.21e8, buoyancy=True)
    203.89681224927565
    >>> Nu_horizontal_plate_VDI(5.54, 3.21e8, buoyancy=False)
    39.16864971535617

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    .. [2] Stewartson, Keith. "On the Free Convection from a Horizontal Plate."
       Zeitschrift Für Angewandte Mathematik Und Physik ZAMP 9, no. 3
       (September 1, 1958): 276-82. https://doi.org/10.1007/BF02033031.
    .. [3] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    '''
    Ra = Pr*Gr
    if buoyancy:
        f2 = (1.0 + (0.322/Pr)**(0.55))**(20.0/11.0)
        if Ra*f2 < 7e4:
            return 0.766*(Ra*f2)**0.2
        else:
            return 0.15*(Ra*f2)**(1.0/3.0)
    else:
        f1 = (1.0 + (0.492/Pr)**(9.0/16.0))**(-16.0/9.0)
        return 0.6*(Ra*f1)**0.2


def Nu_horizontal_plate_Rohsenow(Pr, Gr, buoyancy=True):
    r'''Calculates the Nusselt number for natural convection above a horizontal
    plate according to the Rohsenow, Hartnett, and Cho (1998) [1]_ correlations.
    The plate must be isothermal. Three different equations are used, one each
    for laminar and turbulent for the heat transfer happening at upper surface
    case and one for the case of heat transfer happening at the lower surface.

    The lower surface correlation is recommened for the laminar flow regime.
    The two different sets of correlations are required because if the plate
    is hot, buoyancy lifts the fluid off the plate and enhances free convection
    whereas if the plate is cold, the cold fluid above it settles on it and
    decreases the free convection.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to length, [-]

    Notes
    -----
    The characteristic length suggested for use is as follows, with `a` and
    `b` being the length and width of the plate.

    .. math::
        L = \frac{ab}{2(a+b)}


    Examples
    --------
    >>> Nu_horizontal_plate_Rohsenow(5.54, 3.21e8, buoyancy=True)
    175.91054716322836
    >>> Nu_horizontal_plate_Rohsenow(5.54, 3.21e8, buoyancy=False)
    35.95799244863986

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    Ra = Pr*Gr
    if buoyancy:
        C_tU = 0.14*((1.0 + 0.01707*Pr)/(1.0 + 0.01*Pr))
        C_tV = 0.13*Pr**0.22/(1.0 + 0.61*Pr**0.81)**0.42

        t1 = 1.0 # Ah/A # Heated to non heated area ratio
        t2 = 0.0 # Lf*P/A # Lf vertical distance between lowest and highest point in body
        # P is perimiter, A is area
        Cl = (0.0972 - (0.0157 + 0.462*C_tV)*t1
              + (0.615*C_tV - 0.0548 - 6e-6*Pr)*t2)

        Nu_T = 0.835*Cl*Ra**0.25 # average Cl
        Nu_l = 1.4/(log(1.0 + 1.4/Nu_T))
        Nu_t = C_tU*Ra**(1.0/3.0)

        m = 10.0
        Nu = ((Nu_l)**m + Nu_t**m)**(1.0/m)
        return Nu
    else:
        # No friction/C term
        Nu_T = 0.527*Ra**0.2/(1.0 + (1.9/Pr)**0.9)**(2.0/9.0)
        Nu_l = 2.5/(log(1.0 + 2.5/Nu_T))
        return Nu_l


conv_free_horizontal_plate_all_methods = {
    'McAdams': (Nu_horizontal_plate_McAdams, ('Pr', 'Gr', 'buoyancy')),
    'VDI': (Nu_horizontal_plate_VDI, ('Pr', 'Gr', 'buoyancy')),
    'Rohsenow': (Nu_horizontal_plate_Rohsenow, ('Pr', 'Gr', 'buoyancy')),
}

Nu_free_horizontal_plate_all_methods = ["VDI", "McAdams", "Rohsenow"]


def Nu_free_horizontal_plate_methods(Pr, Gr, buoyancy, L=None, W=None,
                                     check_ranges=True):
    r'''This function returns a list of methods for calculating heat transfer
    coefficient for external free convection from a verical plate.

    Requires at a minimum a fluid's Prandtl number `Pr`, and the Grashof
    number `Gr` for the system fluid, temperatures, and geometry.

    `L` and `W` are not used by any correlations presently, but are included
    for future support.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]
    L : float, optional
        Length of horizontal plate, [m]
    W : float, optional
        Width of the horizontal plate, [m]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `Nu` with the given
        inputs, [-]

    Examples
    --------
    >>> Nu_free_horizontal_plate_methods(0.69, 2.63E9, True)
    ['VDI', 'McAdams', 'Rohsenow']
    '''
    return Nu_free_horizontal_plate_all_methods

def Nu_free_horizontal_plate(Pr, Gr, buoyancy, L=None, W=None,
                             Method=None):
    r'''This function calculates the heat transfer coefficient for external
    free convection from a horizontal plate.

    Requires at a minimum a fluid's Prandtl number `Pr`, and the Grashof
    number `Gr` for the system fluid, temperatures, and geometry.

    `L` and `W` are not used by any correlations presently, but are included
    for future support.

    If no correlation's name is provided as `Method`, the 'VDI' correlation is
    selected.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to fluid properties [-]
    Gr : float
        Grashof number with respect to fluid properties and plate - fluid
        temperature difference [-]
    buoyancy : bool, optional
        Whether or not the plate's free convection is buoyancy assisted (hot
        plate) or not, [-]
    L : float, optional
        Length of horizontal plate, [m]
    W : float, optional
        Width of the horizontal plate, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to plate length, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        conv_free_horizontal_plate_methods

    Examples
    --------
    Turbulent example

    >>> Nu_free_horizontal_plate(5.54, 3.21e8, buoyancy=True)
    203.89681224927565

    >>> Nu_free_horizontal_plate(5.54, 3.21e8, buoyancy=True, Method='McAdams')
    181.73121274384457
    '''
    if Method is None:
        Method2 = "VDI"
    else:
        Method2 = Method

    if Method2 == 'VDI':
        return Nu_horizontal_plate_VDI(Pr=Pr, Gr=Gr, buoyancy=buoyancy)
    if Method2 == 'McAdams':
        return Nu_horizontal_plate_McAdams(Pr=Pr, Gr=Gr, buoyancy=buoyancy)
    if Method2 == 'Rohsenow':
        return Nu_horizontal_plate_Rohsenow(Pr=Pr, Gr=Gr, buoyancy=buoyancy)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")


def Nu_sphere_Churchill(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a sphere
    according to the Churchill [1]_ correlation. Sphere must be isothermal.

    .. math::
        Nu_D=2+\frac{0.589Ra_D^{1/4}} {\left[1+(0.469/Pr)^{9/16}\right]^{4/9}}
        \cdot\left\{1 + \frac{7.44\times 10^{-8}Ra}
        {[1+(0.469/Pr)^{9/16}]^{16/9}}\right\}^{1/12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Although transition from laminar to turbulent is discrete in reality, this
    equation provides a smooth transition in value from laminar to turbulent.
    Checked with the original source.

    Good for Ra < 1E13. Limit of Nu is 2 at low Grashof numbers.

    Examples
    --------
    >>> Nu_sphere_Churchill(.7, 1E7)
    25.670869440317578

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    '''
    Ra = Pr*Gr
    Nu = 2 + (0.589*Ra**0.25/(1 + (0.469/Pr)**(9/16.))**(4/9.)*(
         1 + 7.44E-8*Ra/(1 + (0.469/Pr)**(9/16.))**(16/9.))**(1/12.))
    return Nu


### Vertical cylinders

def Nu_vertical_cylinder_Griffiths_Davis_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.67 Ra_H^{0.25},\; 10^{7} < Ra < 10^{9}

    .. math::
        Nu_H = 0.0782 Ra_H^{0.357}, \; 10^{9} < Ra < 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameter 17.43 cm, length from 4.65 to 263.5 cm. Air as fluid.
    Transition between ranges is not smooth.
    If outside of range, no warning is given.

    Examples
    --------
    >>> Nu_vertical_cylinder_Griffiths_Davis_Morgan(.7, 2E10)
    327.6230596100138

    References
    ----------
    .. [1] Griffiths, Ezer, A. H. Davis, and Great Britain. The Transmission of
       Heat by Radiation and Convection. London: H. M. Stationery off., 1922.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E9 and turbulent is None):
        Nu = 0.0782*Ra**0.357
    else:
        Nu = 0.67*Ra**0.25
    return Nu


def Nu_vertical_cylinder_Jakob_Linke_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.555 Ra_H^{0.25},\; 10^{4} < Ra < 10^{8}

    .. math::
        Nu_H = 0.129 Ra_H^{1/3},\; 10^{8} < Ra < 10^{12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameter 3.5 cm, length from L/D = 4.3. Air as fluid.
    Transition between ranges is not smooth.
    If outside of range, no warning is given. Results are presented rounded in
    [4]_, and the second range is not shown in [3]_.

    Examples
    --------
    >>> Nu_vertical_cylinder_Jakob_Linke_Morgan(.7, 2E10)
    310.90835207860454

    References
    ----------
    .. [1] Jakob, M., and Linke, W., Warmeubergang beim Verdampfen von
       Flussigkeiten an senkrechten und waagerechten Flaschen, Phys. Z.,
       vol. 36, pp. 267-280, 1935.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E8 and turbulent is None):
        Nu = 0.129*Ra**(1/3.)
    else:
        Nu = 0.555*Ra**0.25
    return Nu


def Nu_vertical_cylinder_Carne_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 1.07 Ra_H^{0.28},\; 2\times 10^{6} < Ra < 2\times 10^{8}

    .. math::
        Nu_H = 0.152 Ra_H^{0.38},\; 2\times 10^{8} < Ra < 2\times 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameters 0.475 cm to 7.62 cm, L/D from 8 to 127. Isothermal
    boundary condition was assumed, but not verified. Transition between ranges
    is not smooth. If outside of range, no warning is given. The higher range
    of [1]_ is not shown in [3]_, and the formula for the first is actually for
    the second in [3]_.

    Examples
    --------
    >>> Nu_vertical_cylinder_Carne_Morgan(.7, 2E8)
    204.31470629065677

    References
    ----------
    .. [1] J. B. Carne. "LIX. Heat Loss by Natural Convection from Vertical
       Cylinders." The London, Edinburgh, and Dublin Philosophical Magazine and
       Journal of Science 24, no. 162 (October 1, 1937): 634-53.
       doi:10.1080/14786443708565140.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 2E8 and turbulent is None):
        return 0.152*Ra**0.38
    else:
        return 1.07*Ra**0.28


def Nu_vertical_cylinder_Eigenson_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_,
    presented in [3]_ and in more detail in [4]_.

    .. math::
        Nu_H = 0.48 Ra_H^{0.25},\; 10^{9} < Ra

    .. math::
        Nu_H = 51.5 + 0.0000726 Ra_H^{0.63},\; 10^{9} < Ra < 1.69 \times 10^{10}

    .. math::
        Nu_H = 0.148 Ra_H^{1/3} - 127.6 ,\; 1.69 \times 10^{10} < Ra

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Author presents results as appropriate for both flat plates and cylinders.
    Height of 2.5 m with diameters of 2.4, 7.55, 15, 35, and 50 mm. Another
    experiment of diameter 58 mm and length of 6.5 m was considered.
    Cylinder of diameters 0.475 cm to 7.62 cm, L/D from 8 to 127.Transition
    between ranges is not smooth. If outside of range, no warning is given.
    Formulas are presented similarly in [3]_ and [4]_, but only [4]_ shows
    the transition formula.

    Examples
    --------
    >>> Nu_vertical_cylinder_Eigenson_Morgan(0.7, 2E10)
    230.55946525499715

    References
    ----------
    .. [1] Eigenson L (1940). Les lois gouvernant la transmission de la chaleur
       aux gaz biatomiques par les parois des cylindres verticaux dans le cas
       de convection naturelle. Dokl Akad Nauk SSSR 26:440-444
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1.69E10 and turbulent is None):
        return 0.148*Ra**(1/3.) - 127.6
    elif 1E9 < Ra < 1.69E10 and turbulent is not False:
        return 51.5 + 0.0000726*Ra**0.63
    else:
        return 0.48*Ra**0.25


def Nu_vertical_cylinder_Touloukian_Morgan(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by [2]_, as
    presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.726 Ra_H^{0.25},\; 2\times 10^{8} < Ra < 4\times 10^{10}

    .. math::
        Nu_H = 0.0674 (Gr_H Pr^{1.29})^{1/3},\; 4\times 10^{10} < Ra < 9\times 10^{11}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Cylinder of diameters 2.75 inch, with heights of 6, 18, and 36.25 inch.
    Temperature was controlled via multiple separately controlled heating
    sections. Fluids were water and ethylene-glycol. Transition between ranges
    is not smooth. If outside of range, no warning is given. [2]_, [3]_, and
    [4]_ are in complete agreement about this formulation.

    Examples
    --------
    >>> Nu_vertical_cylinder_Touloukian_Morgan(.7, 2E10)
    249.72879961097854

    References
    ----------
    .. [1] Touloukian, Y. S, George A Hawkins, and Max Jakob. Heat Transfer by
       Free Convection from Heated Vertical Surfaces to Liquids.
       Trans. ASME 70, 13-18 (1948).
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 4E10 and turbulent is None):
        return 0.0674*(Gr*Pr**1.29)**(1/3.)
    else:
        return 0.726*Ra**0.25


def Nu_vertical_cylinder_McAdams_Weiss_Saunders(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ and [2]_ correlated by
    [3]_, as presented in [4]_, [5]_, and [6]_.

    .. math::
        Nu_H = 0.59 Ra_H^{0.25},\; 10^{4} < Ra < 10^{9}

    .. math::
        Nu_H = 0.13 Ra_H^{1/3.},\; 10^{9} < Ra < 10^{12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Transition between ranges is not smooth. If outside of range, no warning is
    given. For ranges under 10^4, a graph is provided, not included here.

    Examples
    --------
    >>> Nu_vertical_cylinder_McAdams_Weiss_Saunders(.7, 2E10)
    313.31849434277973

    References
    ----------
    .. [1] Weise, Rudolf. "Warmeubergang durch freie Konvektion an
       quadratischen Platten." Forschung auf dem Gebiet des Ingenieurwesens
       A 6, no. 6 (November 1935): 281-92. doi:10.1007/BF02592565.
    .. [2] Saunders, O. A. "The Effect of Pressure Upon Natural Convection in
       Air." Proceedings of the Royal Society of London A: Mathematical,
       Physical and Engineering Sciences 157, no. 891 (November 2, 1936):
       278-91. doi:10.1098/rspa.1936.0194.
    .. [3] McAdams, William Henry. Heat Transmission. 3E. Malabar, Fla:
       Krieger Pub Co, 1985.
    .. [4] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [5] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [6] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E9 and turbulent is None):
        return 0.13*Ra**(1/3.)
    else:
        return 0.59*Ra**0.25


def Nu_vertical_cylinder_Kreith_Eckert(Pr, Gr, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_  correlated by
    [2]_, also as presented in [3]_, [4]_, and [5]_.

    .. math::
        Nu_H = 0.555 Ra_H^{0.25},\; 10^{5} < Ra < 10^{9}

    .. math::
        Nu_H = 0.021 Ra_H^{0.4},\; 10^{9} < Ra < 10^{12}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Transition between ranges is not smooth. If outside of range, no warning is
    given.

    Examples
    --------
    >>> Nu_vertical_cylinder_Kreith_Eckert(.7, 2E10)
    240.25393473033196

    References
    ----------
    .. [1] Eckert, E. R. G., Thomas W. Jackson, and United States. Analysis of
       Turbulent Free-Convection Boundary Layer on Flat Plate. National
       Advisory Committee for Aeronautics, no. 2207. Washington, D.C.: National
       Advisoty Committee for Aeronautics, 1950.
    .. [2] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer. Cengage, 2010.
    .. [3] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [4] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [5] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if turbulent or (Ra > 1E9 and turbulent is None):
        return 0.021*Ra**0.4
    else:
        return 0.555*Ra**0.25


def Nu_vertical_cylinder_Hanesian_Kalish_Morgan(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to the results of [1]_ correlated by
    [2]_, also as presented in [3]_ and [4]_.

    .. math::
        Nu_H = 0.48 Ra_H^{0.23},\; 10^{6} < Ra < 10^{8}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    For air and fluoro-carbons. If outside of range, no warning is given.
    Laminar range only!

    Examples
    --------
    >>> Nu_vertical_cylinder_Hanesian_Kalish_Morgan(.7, 1E7)
    18.014150492696604

    References
    ----------
    .. [1] Hanesian, D. and Kalish, R. "Heat Transfer by Natural Convection
       with Fluorocarbon Gases." IEEE Transactions on Parts, Materials and
       Packaging 6, no. 4 (December 1970): 147-148.
       doi:10.1109/TPMP.1970.1136270.
    .. [2] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [3] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [4] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    return 0.48*Ra**0.23


### Vertical cylinders, more complex correlations
def Nu_vertical_cylinder_Al_Arabi_Khamis(Pr, Gr, L, D, turbulent=None):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to [1]_, also as presented in [2]_ and [3]_.

    .. math::
        Nu_H = 2.9Ra_H^{0.25}/Gr_D^{1/12},\; 9.88 \times 10^7 \le Ra_H \le 2.7\times10^{9}

    .. math::
        Nu_H = 0.47 Ra_H^{0.333}/Gr_D^{1/12},\; 2.7 \times 10^9 \le Ra_H \le 2.95\times10^{10}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder height [-]
    L : float
        Length of vertical cylinder, [m]
    D : float
        Diameter of cylinder, [m]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False; leave as None for
        automatic selection, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    For air. Local Nusselt number results also given in [1]_. D from 12.75 to
    51 mm; H from 300 to 2000 mm. Temperature kept constant by steam condensing.

    If outside of range, no warning is given. Applies for range of:

    .. math::
        1.08 \times 10^4 \le Gr_D \le 6.9 \times 10^5

    Examples
    --------
    >>> Nu_vertical_cylinder_Al_Arabi_Khamis(.71, 2E10, 10, 1)
    280.39793209114765

    References
    ----------
    .. [1] Al-Arabi, M., and M. Khamis. "Natural Convection Heat Transfer from
       Inclined Cylinders." International Journal of Heat and Mass Transfer 25,
       no. 1 (January 1982): 3-15. doi:10.1016/0017-9310(82)90229-0.
    .. [2] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [3] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Gr_D = Gr/L**3*D**3
    Ra = Pr*Gr
    if turbulent or (Ra > 2.6E9 and turbulent is None):
        return 0.47*Ra**(1/3.)*Gr_D**(-1/12.)
    else:
        return 2.9*Ra**0.25*Gr_D**(-1/12.)


def Nu_vertical_cylinder_Popiel_Churchill(Pr, Gr, L, D):
    r'''Calculates Nusselt number for natural convection around a vertical
    isothermal cylinder according to [1]_, also  presented in [2]_.

    .. math::
        \frac{Nu}{Nu_{L,fp}} = 1 + B\left[32^{0.5}Gr_L^{-0.25}\frac{L}{D}\right]^C

    .. math::
        B = 0.0571322 + 0.20305 Pr^{-0.43}

    .. math::
        C = 0.9165 - 0.0043Pr^{0.5} + 0.01333\ln Pr + 0.0004809/Pr

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder height [-]
    L : float
        Length of vertical cylinder, [m]
    D : float
        Diameter of cylinder, [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    For 0.01 < Pr < 100. Requires a vertical flat plate correlation.
    Both [2], [3] present a power of 2 instead of 0.5 on the 32 in the equation,
    but the original has the correct form.

    Examples
    --------
    >>> Nu_vertical_cylinder_Popiel_Churchill(0.7, 1E10, 2.5, 1)
    228.89790055149896

    References
    ----------
    .. [1] Popiel, C. O., J. Wojtkowiak, and K. Bober. "Laminar Free Convective
       Heat Transfer from Isothermal Vertical Slender Cylinder." Experimental
       Thermal and Fluid Science 32, no. 2 (November 2007): 607-613.
       doi:10.1016/j.expthermflusci.2007.07.003.
    .. [2] Popiel, Czeslaw O. "Free Convection Heat Transfer from Vertical
       Slender Cylinders: A Review." Heat Transfer Engineering 29, no. 6
       (June 1, 2008): 521-36. doi:10.1080/01457630801891557.
    .. [3] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    B = 0.0571322 + 0.20305*Pr**-0.43
    C = 0.9165 - 0.0043*Pr**0.5 + 0.01333*log(Pr) + 0.0004809/Pr
    Nu_fp = Nu_vertical_plate_Churchill(Pr, Gr)
    return Nu_fp*(1 + B*(32**0.5*Gr**-0.25*L/D)**C)



# Nice Name : (function_call, does_turbulent, does_laminar, transition_Ra, is_only_Pr_Gr)
vertical_cylinder_correlations = {
'Churchill Vertical Plate': (Nu_vertical_plate_Churchill, True, True, None, True),
'Griffiths, Davis, & Morgan': (Nu_vertical_cylinder_Griffiths_Davis_Morgan, True, True, 1.00E+009, True),
'Jakob, Linke, & Morgan': (Nu_vertical_cylinder_Jakob_Linke_Morgan, True, True, 1.00E+008, True),
'Carne & Morgan': (Nu_vertical_cylinder_Carne_Morgan, True, True, 2.00E+008, True),
'Eigenson & Morgan': (Nu_vertical_cylinder_Eigenson_Morgan, True, True, 6.90E+011, True),
'Touloukian & Morgan': (Nu_vertical_cylinder_Touloukian_Morgan, True, True, 4.00E+010, True),
'McAdams, Weiss & Saunders': (Nu_vertical_cylinder_McAdams_Weiss_Saunders, True, True, 1.00E+009, True),
'Kreith & Eckert': (Nu_vertical_cylinder_Kreith_Eckert, True, True, 1.00E+009, True),
'Hanesian, Kalish & Morgan': (Nu_vertical_cylinder_Hanesian_Kalish_Morgan, False, True, 1.00E+008, True),
'Al-Arabi & Khamis': (Nu_vertical_cylinder_Al_Arabi_Khamis, True, True, 2.60E+009, False),
'Popiel & Churchill': (Nu_vertical_cylinder_Popiel_Churchill, False, True, 1.00E+009, False),
}

def Nu_vertical_cylinder_methods(Pr, Gr, L=None, D=None, check_ranges=True):
    r'''This function returns a list of correlation names for free convetion
    to a vertical cylinder.

    The functions returned are 'Popiel & Churchill' for fully defined geometries,
    and 'McAdams, Weiss & Saunders' otherwise.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder height [-]
    L : float, optional
        Length of vertical cylinder, [m]
    D : float, optional
        Diameter of cylinder, [m]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]


    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `Nu` with the given
        inputs

    Examples
    --------
    >>> Nu_vertical_cylinder_methods(0.72, 1E7)[0]
    'McAdams, Weiss & Saunders'
    '''
    if L is None or D is None:
        return ['McAdams, Weiss & Saunders', 'Churchill Vertical Plate',
                'Griffiths, Davis, & Morgan', 'Jakob, Linke, & Morgan', 'Carne & Morgan',
                'Eigenson & Morgan', 'Touloukian & Morgan', 'Kreith & Eckert', 'Hanesian, Kalish & Morgan']
    else:
        return ['Popiel & Churchill', 'Churchill Vertical Plate', 'Griffiths, Davis, & Morgan',
                'Jakob, Linke, & Morgan', 'Carne & Morgan', 'Eigenson & Morgan', 'Touloukian & Morgan',
                'McAdams, Weiss & Saunders', 'Kreith & Eckert', 'Hanesian, Kalish & Morgan',
                'Al-Arabi & Khamis']


def Nu_vertical_cylinder(Pr, Gr, L=None, D=None, Method=None):
    r'''This function handles choosing which vertical cylinder free convection
    correlation is used. Generally this is used by a helper class, but can be
    used directly. Will automatically select the correlation to use if none is
    provided; returns None if insufficient information is provided.

    Preferred functions are 'Popiel & Churchill' for fully defined geometries,
    and 'McAdams, Weiss & Saunders' otherwise.

    Examples
    --------
    >>> Nu_vertical_cylinder(0.72, 1E7)
    30.562236756513943

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder height [-]
    L : float, optional
        Length of vertical cylinder, [m]
    D : float, optional
        Diameter of cylinder, [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        vertical_cylinder_correlations
    '''
    if Method is None:
        if L is None or D is None:
            Method2 = 'McAdams, Weiss & Saunders'
        else:
            Method2 = 'Popiel & Churchill'
    else:
        Method2 = Method

    if Method2 == 'Churchill Vertical Plate':
        return Nu_vertical_plate_Churchill(Pr=Pr, Gr=Gr)
    elif Method2 == 'Griffiths, Davis, & Morgan':
        return Nu_vertical_cylinder_Griffiths_Davis_Morgan(Pr=Pr, Gr=Gr)
    elif Method2 == 'Jakob, Linke, & Morgan':
        return Nu_vertical_cylinder_Jakob_Linke_Morgan(Pr=Pr, Gr=Gr)
    elif Method2 == 'Carne & Morgan':
        return Nu_vertical_cylinder_Carne_Morgan(Pr=Pr, Gr=Gr)
    elif Method2 == 'Eigenson & Morgan':
        return Nu_vertical_cylinder_Eigenson_Morgan(Pr=Pr, Gr=Gr)
    elif Method2 == 'Touloukian & Morgan':
        return Nu_vertical_cylinder_Touloukian_Morgan(Pr=Pr, Gr=Gr)
    elif Method2 == 'McAdams, Weiss & Saunders':
        return Nu_vertical_cylinder_McAdams_Weiss_Saunders(Pr=Pr, Gr=Gr)
    elif Method2 == 'Kreith & Eckert':
        return Nu_vertical_cylinder_Kreith_Eckert(Pr=Pr, Gr=Gr)
    elif Method2 == 'Hanesian, Kalish & Morgan':
        return Nu_vertical_cylinder_Hanesian_Kalish_Morgan(Pr=Pr, Gr=Gr)

    elif Method2 == 'Al-Arabi & Khamis':
        return Nu_vertical_cylinder_Al_Arabi_Khamis(Pr=Pr, Gr=Gr, L=L, D=D)
    elif Method2 == 'Popiel & Churchill':
        return Nu_vertical_cylinder_Popiel_Churchill(Pr=Pr, Gr=Gr, L=L, D=D)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")

#import matplotlib.pyplot as plt
#import numpy as np
##L, D = 1.5, 0.1
#Pr, Gr = 0.72, 1E8
#methods = Nu_vertical_cylinder_methods(Pr, Gr)
#Grs = np.logspace(2, 12, 10000)
#
#for method in methods:
#    Nus = [Nu_vertical_cylinder(Pr=Pr, Gr=i, Method=method) for i in Grs]
#    plt.loglog(Grs, Nus, label=method)
#plt.legend()
#plt.show()


### Horizontal Cylinders

def Nu_horizontal_cylinder_Churchill_Chu(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a horizontal
    cylinder according to the Churchill-Chu [1]_ correlation, also presented in
    [2]_. Cylinder must be isothermal; an alternate expression exists for
    constant heat flux.

    .. math::
        Nu_{D}=\left[0.60+\frac{0.387Ra_{D}^{1/6}}
        {[1+(0.559/Pr)^{9/16}]^{8/27}}\right]^2

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with respect to cylinder diameter, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Although transition from laminar to turbulent is discrete in reality, this
    equation provides a smooth transition in value from laminar to turbulent.
    Checked with the original source, which has its powers unsimplified but
    is equivalent.

    [1]_ recommends 1E-5 as the lower limit for Ra, but no upper limit. [2]_
    suggests an upper limit of 1E12.

    Examples
    --------
    From [2]_, Example 9.2, matches:

    >>> Nu_horizontal_cylinder_Churchill_Chu(0.69, 2.63E9)
    139.13493970073597

    References
    ----------
    .. [1] Churchill, Stuart W., and Humbert H. S. Chu. "Correlating Equations
       for Laminar and Turbulent Free Convection from a Horizontal Cylinder."
       International Journal of Heat and Mass Transfer 18, no. 9
       (September 1975): 1049-53. doi:10.1016/0017-9310(75)90222-7.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Ra = Pr*Gr
    return (0.6 + 0.387*Ra**(1/6.)/(1. + (0.559/Pr)**(9/16.))**(8/27.))**2


def Nu_horizontal_cylinder_Kuehn_Goldstein(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a horizontal
    cylinder according to the Kuehn-Goldstein [1]_ correlation, also shown in
    [2]_. Cylinder must be isothermal.

    .. math::
        \frac{2}{Nu_D} = \ln\left[1 + \frac{2}{\left[\left\{0.518Ra_D^{0.25}
        \left[1 + \left(\frac{0.559}{Pr}\right)^{3/5}\right]^{-5/12}
        \right\}^{15} + (0.1Ra_D^{1/3})^{15}\right]^{1/15}}\right]

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to film temperature [-]
    Gr : float
        Grashof number with respect to cylinder diameter, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    [1]_ suggests this expression is valid for all cases except low-Pr fluids.
    [2]_ suggests no restrictions.

    Examples
    --------
    >>> Nu_horizontal_cylinder_Kuehn_Goldstein(0.69, 2.63E9)
    122.99323525628186

    References
    ----------
    .. [1] Kuehn, T. H., and R. J. Goldstein. "Correlating Equations for
       Natural Convection Heat Transfer between Horizontal Circular Cylinders."
       International Journal of Heat and Mass Transfer 19, no. 10
       (October 1976): 1127-34. doi:10.1016/0017-9310(76)90145-9
    .. [2] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    return 2./log(1 + 2./((0.518*Ra**0.25*(1. + (0.559/Pr)**0.6)**(-5/12.))**15
                  + (0.1*Ra**(1/3.))**15)**(1/15.))


def Nu_horizontal_cylinder_Morgan(Pr, Gr):
    r'''Calculates Nusselt number for natural convection around a horizontal
    cylinder according to the Morgan [1]_ correlations, a product of a very
    large review of the literature. Sufficiently common as to be shown in [2]_.
    Cylinder must be isothermal.

    .. math::
        Nu_D = C Ra_D^n

    +----------+----------+-------+-------+
    |  Gr min  |  Gr max  |  C    |  n    |
    +==========+==========+=======+=======+
    | 10E-10   |  10E-2   | 0.675 | 0.058 |
    +----------+----------+-------+-------+
    | 10E-2    |  10E2    | 1.02  | 0.148 |
    +----------+----------+-------+-------+
    | 10E2     |  10E4    | 0.850 | 0.188 |
    +----------+----------+-------+-------+
    | 10E4     |  10E7    | 0.480 | 0.250 |
    +----------+----------+-------+-------+
    | 10E7     |  10E12   | 0.125 | 0.333 |
    +----------+----------+-------+-------+

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to film temperature [-]
    Gr : float
        Grashof number with respect to cylinder diameter, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Notes
    -----
    Most comprehensive review with a new proposed equation to date.
    Discontinuous among the jumps in range. Blindly runs outside if upper and
    lower limits without warning.

    Examples
    --------
    >>> Nu_horizontal_cylinder_Morgan(0.69, 2.63E9)
    151.3881997228419

    References
    ----------
    .. [1] Morgan, V.T., The Overall Convective Heat Transfer from Smooth
       Circular Cylinders, in Advances in Heat Transfer, eds. T.F. Irvin and
       J.P. Hartnett, V 11, 199-264, 1975.
    .. [2] Boetcher, Sandra K. S. "Natural Convection Heat Transfer From
       Vertical Cylinders." In Natural Convection from Circular Cylinders,
       23-42. Springer, 2014.
    '''
    Ra = Pr*Gr
    if Ra < 1E-2:
        C, n = 0.675, 0.058
    elif Ra < 1E2:
        C, n = 1.02, 0.148
    elif Ra < 1E4:
        C, n = 0.850, 0.188
    elif Ra < 1E7:
        C, n = 0.480, 0.250
    else:
        # up to 1E12
        C, n = 0.125, 0.333
    return C*Ra**n


horizontal_cylinder_correlations = {
'Churchill-Chu': (Nu_horizontal_cylinder_Churchill_Chu),
'Kuehn & Goldstein':  (Nu_horizontal_cylinder_Kuehn_Goldstein),
'Morgan': (Nu_horizontal_cylinder_Morgan)
}

def Nu_horizontal_cylinder_methods(Pr, Gr, check_ranges=True):
    r'''This function returns a list of correlation names for free convetion
    to a horizontal cylinder.

    Preferred functions are 'Morgan' when discontinuous results are acceptable
    and 'Churchill-Chu' otherwise.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to film temperature [-]
    Gr : float
        Grashof number with respect to cylinder diameter, [-]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list[str]
        List of methods which can be used to calculate `Nu` with the given
        inputs

    Examples
    --------
    >>> Nu_horizontal_cylinder_methods(0.72, 1E7)[0]
    'Morgan'
    '''
    return ['Morgan', 'Churchill-Chu', 'Kuehn & Goldstein']

def Nu_horizontal_cylinder(Pr, Gr, Method=None):
    r'''This function handles choosing which horizontal cylinder free convection
    correlation is used. Generally this is used by a helper class, but can be
    used directly. Will automatically select the correlation to use if none is
    provided; returns None if insufficient information is provided.

    Preferred functions are 'Morgan' when discontinuous results are acceptable
    and 'Churchill-Chu' otherwise.

    Parameters
    ----------
    Pr : float
        Prandtl number with respect to film temperature [-]
    Gr : float
        Grashof number with respect to cylinder diameter, [-]

    Returns
    -------
    Nu : float
        Nusselt number with respect to cylinder diameter, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        horizontal_cylinder_correlations

    Notes
    -----
    All fluid properties should be evaluated at the film temperature, the
    average between the outer surface temperature of the solid, and the fluid
    temperature far away from the heat transfer interface - normally the same
    as the temperature before any cooling or heating occurs.

    .. math::
        T_f = (T_{\text{surface}} + T_\infty)/2

    Examples
    --------
    >>> Nu_horizontal_cylinder(0.72, 1E7)
    24.864192615468973
    '''
    if Method is None:
        Method2 = 'Morgan'
    else:
        Method2 = Method
    if Method2 == 'Churchill-Chu':
        return Nu_horizontal_cylinder_Churchill_Chu(Pr=Pr, Gr=Gr)
    elif Method2 == 'Kuehn & Goldstein':
        return Nu_horizontal_cylinder_Kuehn_Goldstein(Pr=Pr, Gr=Gr)
    elif Method2 == 'Morgan':
        return Nu_horizontal_cylinder_Morgan(Pr=Pr, Gr=Gr)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")


#import matplotlib.pyplot as plt
#import numpy as np
#Pr, Gr = 0.72, 1E8
#methods = Nu_horizontal_cylinder_methods(Pr, Gr)
#Grs = np.logspace(-2, 2.5, 10000)
#
#for method in methods:
#    Nus = [Nu_horizontal_cylinder(Pr=Pr, Gr=i, Method=method) for i in Grs]
#    plt.semilogx(Grs, Nus, label=method)
#plt.legend()
#plt.show()


def Nu_coil_Xin_Ebadian(Pr, Gr, horizontal=False):
    r'''Calculates Nusselt number for natural convection around a vertical
    or horizontal helical coil suspended in a fluid without
    forced convection.

    For horizontal cases:

    .. math::
        Nu_D = 0.318 Ra_D^{0.293},\; 5 \times {10}^{3} < Ra < 1 \times {10}^5

    For vertical cases:

    .. math::
        Nu_D = 0.290 Ra_D^{0.293},\; 5 \times {10}^{3} < Ra < 1 \times {10}^5

    Parameters
    ----------
    Pr : float
        Prandtl number calculated with the film temperature -
        wall and temperature very far from the coil average, [-]
    Gr : float
        Grashof number calculated with the film temperature -
        wall and temperature very far from the coil average,
        and using the outer diameter of the coil [-]
    horizontal : bool, optional
        Whether the coil is horizontal or vertical, [-]

    Returns
    -------
    Nu : float
        Nusselt number using the outer diameter of the coil
        and the film temperature, [-]

    Notes
    -----
    This correlation is also reviewed in [2]_.

    Examples
    --------
    >>> Nu_coil_Xin_Ebadian(0.7, 2E4, horizontal=False)
    4.755689726250451
    >>> Nu_coil_Xin_Ebadian(0.7, 2E4, horizontal=True)
    5.2148597687849785

    References
    ----------
    .. [1] Xin, R. C., and M. A. Ebadian. "Natural Convection Heat Transfer
       from Helicoidal Pipes." Journal of Thermophysics and Heat Transfer 10,
       no. 2 (1996): 297-302.
    .. [2] Prabhanjan, Devanahalli G., Timothy J. Rennie, and G. S. Vijaya
       Raghavan. "Natural Convection Heat Transfer from Helical Coiled Tubes."
       International Journal of Thermal Sciences 43, no. 4 (April 1, 2004):
       359-65.
    '''
    Ra = Pr*Gr
    if horizontal:
        return 0.318*Ra**0.293
    else:
        return 0.290*Ra**0.293


def Nu_vertical_plate_laminar_Cengel(Ra):
    r'''Calculates the average Nusselt number for laminar natural convection
    over a vertical flat plate with uniform surface temperature, according to
    Eq. 9-19 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.59 Ra_L^{1/4} \quad (10^4 \le Ra_L \le 10^9)

    Parameters
    ----------
    Ra : float
        Rayleigh number based on plate height L [-]

    Returns
    -------
    Nu : float
        Average Nusselt number with respect to plate height, [-]

    Notes
    -----
    Valid for laminar flow in the range :math:`10^4 \le Ra_L \le 10^9`.
    All fluid properties should be evaluated at the film temperature
    :math:`T_f = (T_s + T_\infty)/2`.

    Examples
    --------
    Example 9-2 from [1]_:
    >>> Nu_vertical_plate_laminar_Cengel(7.649e8) # doctest: +ELLIPSIS
    98.11...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-19, p. 542, Table 9-1.
    '''
    if Ra < 1e4 or Ra > 1e9:
        warnings.warn("Rayleigh number Ra={:.3e} is outside recommended range 10^4 <= Ra <= 10^9 for laminar vertical plate correlation (Eq. 9-19).".format(Ra), RuntimeWarning)
    return 0.59 * Ra**0.25


def Nu_vertical_plate_turbulent_Cengel(Ra):
    r'''Calculates the average Nusselt number for turbulent natural convection
    over a vertical flat plate with uniform surface temperature, according to
    Eq. 9-20 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = 0.10 Ra_L^{1/3} \quad (10^{10} \le Ra_L \le 10^{13})

    Parameters
    ----------
    Ra : float
        Rayleigh number based on plate height L [-]

    Returns
    -------
    Nu : float
        Average Nusselt number with respect to plate height, [-]

    Notes
    -----
    Valid for turbulent flow in the range :math:`10^{10} \le Ra_L \le 10^{13}`.
    All fluid properties should be evaluated at the film temperature
    :math:`T_f = (T_s + T_\infty)/2`.

    Examples
    --------
    >>> Nu_vertical_plate_turbulent_Cengel(1e11) # doctest: +ELLIPSIS
    464.158...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-20, p. 542, Table 9-1.
    '''
    if Ra < 1e10 or Ra > 1e13:
        warnings.warn("Rayleigh number Ra={:.3e} is outside recommended range 10^10 <= Ra <= 10^13 for turbulent vertical plate correlation (Eq. 9-20).".format(Ra), RuntimeWarning)
    return 0.10 * Ra**(1.0/3.0)


def Nu_vertical_plate_Cengel(Pr, Gr, method='Churchill'):
    r'''Calculates the average Nusselt number for natural convection over an
    isothermal vertical flat plate according to Çengel & Ghajar (5th Ed) [1]_.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number based on plate height L [-]
    method : str, optional
        Correlation method to use:
        - 'Churchill': Churchill & Chu (Eq. 9-21), comprehensive relation for all Ra.
        - 'simple': Simple empirical power laws (Eq. 9-19 for laminar, Eq. 9-20 for turbulent).

    Returns
    -------
    Nu : float
        Average Nusselt number with respect to plate height, [-]

    Examples
    --------
    From Example 9-2 in [1]_:
    >>> Nu_vertical_plate_Cengel(0.7202, 1.062066e9, method='Churchill') # doctest: +ELLIPSIS
    113.3...
    >>> Nu_vertical_plate_Cengel(0.7202, 1.062066e9, method='simple') # doctest: +ELLIPSIS
    98.11...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Table 9-1, p. 542.
    '''
    Ra = Pr * Gr
    m = method.lower()
    if m in ('churchill', 'complex'):
        return Nu_vertical_plate_Churchill(Pr, Gr)
    elif m == 'simple':
        if Ra < 1e9:
            return Nu_vertical_plate_laminar_Cengel(Ra)
        else:
            return Nu_vertical_plate_turbulent_Cengel(Ra)
    else:
        raise ValueError("Method '{}' not recognized. Use 'Churchill' or 'simple'.".format(method))


def Nu_inclined_plate_Cengel(Pr, Gr, theta=0.0, hot_surface_facing='up', is_hot_plate=True, method='Churchill'):
    r'''Calculates the average Nusselt number for natural convection over an
    inclined flat plate according to Çengel & Ghajar (5th Ed) [1]_.

    When the boundary layer remains intact (the lower surface of a hot plate
    or the upper surface of a cold plate), the Nusselt number is determined
    from the vertical plate relations by replacing gravitational acceleration
    `g` with `g*cos(theta)` for :math:`0 \le \theta \le 60^\circ`, where `theta`
    is the angle with the vertical.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number with standard gravity `g` [-]
    theta : float, optional
        Angle of inclination with the vertical, in degrees [deg] (0 <= theta <= 60)
    hot_surface_facing : str, optional
        'up' or 'down', orientation of the hotter surface [-]
    is_hot_plate : bool, optional
        True if the plate is hotter than the ambient fluid, False if colder [-]
    method : str, optional
        'Churchill' or 'simple' for vertical plate formulation [-]

    Returns
    -------
    Nu : float
        Average Nusselt number with respect to plate length, [-]

    Notes
    -----
    Valid for :math:`0 \le \theta \le 60^\circ` with the vertical.

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Table 9-1, p. 542 and Section 9-3, p. 544.
    '''
    if theta < 0.0 or theta > 60.0:
        warnings.warn("Angle theta={:.1f} deg exceeds recommended range [0, 60] deg for inclined plate correlation.".format(theta), RuntimeWarning)

    cos_theta = max(0.0, cos(radians(theta)))
    Gr_eff = Gr * cos_theta
    return Nu_vertical_plate_Cengel(Pr, Gr_eff, method=method)


def Nu_horizontal_plate_Cengel(Pr, Gr, hot_surface_facing='up', is_hot_plate=True, flow_regime=None):
    r'''Calculates the average Nusselt number for natural convection over an
    isothermal horizontal flat plate according to Çengel & Ghajar (5th Ed) [1]_.

    Characteristic length is defined as :math:`L_c = A_s / p`, where :math:`A_s`
    is plate surface area and :math:`p` is perimeter.

    (a) Upper surface of a hot plate (or lower surface of a cold plate):
        .. math::
            Nu = 0.54 Ra_L^{1/4} \quad (10^4 \le Ra_L \le 10^7) \quad \text{[Eq. 9-22]} \\
            Nu = 0.15 Ra_L^{1/3} \quad (10^7 \le Ra_L \le 10^{11}) \quad \text{[Eq. 9-23]}

    (b) Lower surface of a hot plate (or upper surface of a cold plate):
        .. math::
            Nu = 0.27 Ra_L^{1/4} \quad (10^5 \le Ra_L \le 10^{11}) \quad \text{[Eq. 9-24]}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number evaluated using characteristic length Lc = As/p [-]
    hot_surface_facing : str, optional
        'up' if the heated surface faces upward, 'down' if downward [-]
    is_hot_plate : bool, optional
        True if plate temperature Ts > T_inf, False if Ts < T_inf [-]
    flow_regime : str, optional
        Explicit regime selection for the upper hot surface:
        - None: Automatically choose Eq. 9-22 for Ra <= 1e7, Eq. 9-23 for Ra > 1e7.
        - 'laminar': Force Eq. 9-22 (as in Çengel Example 9-2).
        - 'turbulent': Force Eq. 9-23.

    Returns
    -------
    Nu : float
        Average Nusselt number based on Lc = As/p, [-]

    Examples
    --------
    Example 9-2 (b) and (c) from [1]_:
    >>> Nu_horizontal_plate_Cengel(0.7202, 1.65926e7, hot_surface_facing='up', is_hot_plate=True, flow_regime='laminar') # doctest: +ELLIPSIS
    31.74...
    >>> Nu_horizontal_plate_Cengel(0.7202, 1.65926e7, hot_surface_facing='up', is_hot_plate=True) # doctest: +ELLIPSIS
    34.29...
    >>> Nu_horizontal_plate_Cengel(0.7202, 1.65926e7, hot_surface_facing='down', is_hot_plate=True) # doctest: +ELLIPSIS
    15.87...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Table 9-1, Eqs. 9-22, 9-23, 9-24, p. 542.
    '''
    Ra = Pr * Gr
    assisted = (is_hot_plate and hot_surface_facing.lower() == 'up') or (not is_hot_plate and hot_surface_facing.lower() == 'down')

    if assisted:
        if Ra < 1e4 or Ra > 1e11:
            warnings.warn("Ra={:.3e} is outside recommended range 10^4 <= Ra <= 10^11 for hot surface facing up.".format(Ra), RuntimeWarning)
        if flow_regime is not None:
            regime = flow_regime.lower()
            if regime == 'laminar':
                return 0.54 * Ra**0.25
            elif regime == 'turbulent':
                return 0.15 * Ra**(1.0/3.0)
            else:
                raise ValueError("flow_regime '{}' not recognized. Use 'laminar', 'turbulent', or None.".format(flow_regime))
        if Ra <= 1e7:
            return 0.54 * Ra**0.25
        else:
            return 0.15 * Ra**(1.0/3.0)
    else:
        if Ra < 1e5 or Ra > 1e11:
            warnings.warn("Ra={:.3e} is outside recommended range 10^5 <= Ra <= 10^11 for hot surface facing down.".format(Ra), RuntimeWarning)
        return 0.27 * Ra**0.25


def is_vertical_cylinder_plate_like(L, D, Gr):
    r'''Evaluates whether a vertical cylinder can be modeled as a vertical
    flat plate by checking if boundary layer curvature effects are negligible,
    according to Eq. 9-28 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        D \ge \frac{35 L}{Gr_L^{1/4}}

    Parameters
    ----------
    L : float
        Height of the vertical cylinder, [m]
    D : float
        Outer diameter of the vertical cylinder, [m]
    Gr : float
        Grashof number with respect to height L, [-]

    Returns
    -------
    plate_like : bool
        True if cylinder curvature effects are negligible and vertical plate
        relations can be applied; False if cylinder must be treated as slender.

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-28, p. 543, Table 9-1.
    '''
    if Gr <= 0 or L <= 0:
        return False
    D_crit = 35.0 * L / (Gr**0.25)
    return D >= D_crit


def Nu_vertical_plate_isoflux_Cengel(Pr, Gr_star):
    r'''Calculates local Nusselt number for laminar natural convection along
    a vertical flat plate subjected to uniform surface heat flux (isoflux),
    according to Eq. 9-27 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu_x = 0.60 (Ra_x^*)^{1/5} \quad (10^5 < Ra_x^* < 10^{11})

    where :math:`Ra_x^* = Gr_x^* Pr = \frac{g \beta q_s x^4}{k \nu^2} Pr`.

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr_star : float
        Modified Grashof number based on uniform heat flux qs and position x [-]

    Returns
    -------
    Nu_x : float
        Local Nusselt number at position x, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-27, p. 543.
    '''
    Ra_star = Gr_star * Pr
    if Ra_star < 1e5 or Ra_star > 1e11:
        warnings.warn("Modified Rayleigh number Ra_star={:.3e} is outside recommended range 10^5 < Ra_star < 10^11.".format(Ra_star), RuntimeWarning)
    return 0.60 * Ra_star**0.20


def Ra_vertical_channel(Pr, Gr, S, L):
    r'''Calculates the channel Rayleigh number for vertical parallel plates
    of spacing S and height L, according to Eq. 9-30 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Ra_S = \frac{g \beta (T_s - T_\infty) S^3}{\nu^2} Pr = Ra_L \frac{S^3}{L^3}

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    Gr : float
        Grashof number based on channel spacing S [-]
    S : float
        Plate spacing (channel gap), [m]
    L : float
        Plate height, [m]

    Returns
    -------
    Ra_S : float
        Channel Rayleigh number based on spacing S, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-30, p. 548.
    '''
    return Pr * Gr


def Nu_vertical_channel_isothermal_Bar_Cohen(Ra_S, S, L):
    r'''Calculates average Nusselt number for natural convection cooling of
    isothermal vertical parallel plates (e.g. heat sink fins), according to
    Bar-Cohen and Rohsenow (1984) and Eq. 9-31 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = \frac{h S}{k} = \left[ \frac{576}{(Ra_S S / L)^2} + \frac{2.873}{(Ra_S S / L)^{0.5}} \right]^{-0.5}

    Parameters
    ----------
    Ra_S : float
        Rayleigh number based on plate spacing S [-]
    S : float
        Plate spacing (gap between fins), [m]
    L : float
        Plate height, [m]

    Returns
    -------
    Nu : float
        Average Nusselt number with respect to spacing S, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-31, p. 548.
    .. [2] Bar-Cohen, A., and W. M. Rohsenow. "Thermally Optimum Spacing of
       Vertical Natural Convection Cooled Parallel Plates." Journal of Heat
       Transfer 106 (1984): 116-123.
    '''
    param = Ra_S * S / L
    if param <= 0:
        return 0.0
    return (576.0 / (param * param) + 2.873 / (param**0.5))**(-0.5)


def optimum_fin_spacing_isothermal(L, Ra_L):
    r'''Calculates the optimum fin spacing for natural convection cooling
    of an array of vertical rectangular fins (heat sink) maintained at
    uniform surface temperature, according to Eq. 9-32 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        S_{opt} = 2.714 \frac{L}{Ra_L^{0.25}}

    Parameters
    ----------
    L : float
        Height of the fins in the vertical direction, [m]
    Ra_L : float
        Rayleigh number based on fin height L [-]

    Returns
    -------
    S_opt : float
        Optimum fin spacing, [m]

    Examples
    --------
    Example 9-3 from [1]_:
    >>> optimum_fin_spacing_isothermal(0.18, 1.845e7) # doctest: +ELLIPSIS
    0.00745...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-32, p. 549.
    '''
    if Ra_L <= 0:
        raise ValueError("Rayleigh number Ra_L must be positive.")
    return 2.714 * L / (Ra_L**0.25)


def Nu_optimum_fin_spacing():
    r'''Returns the constant Nusselt number at optimum fin spacing for an
    isothermal vertical heat sink, according to Eq. 9-33 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu = \frac{h S_{opt}}{k} = 1.307

    Returns
    -------
    Nu : float
        Nusselt number at optimum fin spacing (1.307) [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-33, p. 549.
    '''
    return 1.307


def Q_finned_surface_natural(h, n, L, H, Ts, Tinf):
    r'''Calculates total natural convection heat transfer rate from a vertical
    finned heat sink with n fins of height L and fin profile height/width H,
    according to Eq. 9-34 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \dot{Q} = h (2 n L H) (T_s - T_\infty)

    Parameters
    ----------
    h : float
        Convection heat transfer coefficient, [W/(m^2*K)]
    n : int or float
        Number of fins on the heat sink [-]
    L : float
        Height of the fins in the vertical direction, [m]
    H : float
        Protrusion height (depth) of the fins, [m]
    Ts : float
        Surface temperature of the fins, [K] or [deg C]
    Tinf : float
        Ambient fluid temperature, [K] or [deg C]

    Returns
    -------
    Q : float
        Total heat transfer rate by natural convection, [W]

    Examples
    --------
    Example 9-3 from [1]_:
    >>> Q_finned_surface_natural(4.863, 14, 0.18, 0.024, 80.0, 30.0) # doctest: +ELLIPSIS
    29.41...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-34, p. 549.
    '''
    return h * (2.0 * n * L * H) * (Ts - Tinf)


def Ra_star_vertical_channel(Pr, g, beta, qs, S, k, nu):
    r'''Calculates the modified Rayleigh number for isoflux vertical channels
    (such as vertical printed circuit boards), according to Eq. 9-35 in
    Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Ra_S^* = \frac{g \beta q_s S^4}{k \nu^2} Pr

    Parameters
    ----------
    Pr : float
        Prandtl number [-]
    g : float
        Gravitational acceleration, [m/s^2]
    beta : float
        Thermal expansion coefficient, [1/K]
    qs : float
        Uniform heat flux from the board surface, [W/m^2]
    S : float
        Plate spacing (channel width), [m]
    k : float
        Thermal conductivity of fluid, [W/(m*K)]
    nu : float
        Kinematic viscosity of fluid, [m^2/s]

    Returns
    -------
    Ra_star_S : float
        Modified Rayleigh number based on spacing S, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-35, p. 549.
    '''
    return (g * beta * qs * (S**4) / (k * nu**2)) * Pr


def Nu_vertical_channel_isoflux_Bar_Cohen(Ra_star_S, S, L):
    r'''Calculates the Nusselt number at the upper edge of vertical parallel
    plates under uniform heat flux (e.g. PCBs), where maximum temperature occurs,
    according to Bar-Cohen and Rohsenow (1984) and Eq. 9-36 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu_L = \frac{h_L S}{k} = \left[ \frac{48}{Ra_S^* S / L} + \frac{2.51}{(Ra_S^* S / L)^{0.4}} \right]^{-0.5}

    Parameters
    ----------
    Ra_star_S : float
        Modified Rayleigh number based on plate spacing S [-]
    S : float
        Spacing between adjacent plates, [m]
    L : float
        Height of the plates in the vertical direction, [m]

    Returns
    -------
    Nu_L : float
        Nusselt number at the upper edge based on spacing S, [-]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-36, p. 549.
    '''
    param = Ra_star_S * S / L
    if param <= 0:
        return 0.0
    return (48.0 / param + 2.51 / (param**0.4))**(-0.5)


def optimum_spacing_isoflux(k, nu, L, g, beta, qs, Pr):
    r'''Calculates the thermally optimum plate spacing for natural convection
    cooling of vertical parallel plates under uniform heat flux (e.g., PCBs),
    according to Eq. 9-37 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        S_{opt} = 2.12 \left( \frac{k \nu^2 L}{g \beta q_s Pr} \right)^{0.2}

    Parameters
    ----------
    k : float
        Thermal conductivity of fluid, [W/(m*K)]
    nu : float
        Kinematic viscosity of fluid, [m^2/s]
    L : float
        Height of the plates in the vertical direction, [m]
    g : float
        Gravitational acceleration, [m/s^2]
    beta : float
        Thermal expansion coefficient, [1/K]
    qs : float
        Uniform heat flux from plate surface, [W/m^2]
    Pr : float
        Prandtl number [-]

    Returns
    -------
    S_opt : float
        Optimum plate spacing, [m]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-37, p. 550.
    '''
    denom = g * beta * qs * Pr
    if denom <= 0:
        raise ValueError("Parameters produce non-positive denominator.")
    return 2.12 * ((k * (nu**2) * L) / denom)**0.20


def Q_PCBs_natural(qs, n, L, H):
    r'''Calculates total heat transfer rate from an array of vertical
    printed circuit boards (PCBs) dissipating uniform heat flux,
    according to Eq. 9-38 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        \dot{Q} = q_s (2 n L H)

    Parameters
    ----------
    qs : float
        Uniform heat flux, [W/m^2]
    n : int or float
        Number of boards [-]
    L : float
        Board height, [m]
    H : float
        Board length/depth, [m]

    Returns
    -------
    Q : float
        Total heat dissipation rate, [W]

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-38, p. 550.
    '''
    return qs * (2.0 * n * L * H)


def Nu_mixed_convection(Nu_forced, Nu_natural, flow_type='assisting', n=3.0):
    r'''Calculates combined Nusselt number for mixed convection (simultaneous
    natural and forced convection), according to Eq. 9-66 in Çengel & Ghajar (5th Ed) [1]_.

    .. math::
        Nu_{combined} = (Nu_{forced}^n \pm Nu_{natural}^n)^{1/n}

    where the plus sign is used for assisting and transverse flows, and the
    minus sign is used for opposing flows.

    Parameters
    ----------
    Nu_forced : float
        Nusselt number for pure forced convection [-]
    Nu_natural : float
        Nusselt number for pure natural convection [-]
    flow_type : str, optional
        One of 'assisting', 'opposing', or 'transverse' [-]
    n : float, optional
        Correlation exponent; n = 3 for vertical surfaces, larger (3 to 4)
        for horizontal surfaces. Default is 3.0 [-]

    Returns
    -------
    Nu_combined : float
        Combined Nusselt number for mixed convection, [-]

    Examples
    --------
    Example 9-7 from [1]_:
    >>> Nu_mixed_convection(42.14, 36.46, flow_type='assisting', n=3.0) # doctest: +ELLIPSIS
    49.77...
    >>> Nu_mixed_convection(42.14, 36.46, flow_type='opposing', n=3.0) # doctest: +ELLIPSIS
    29.77...

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Eq. 9-66, p. 563.
    '''
    flow = flow_type.lower()
    if flow in ('assisting', 'transverse'):
        return (Nu_forced**n + Nu_natural**n)**(1.0 / n)
    elif flow == 'opposing':
        diff = Nu_forced**n - Nu_natural**n
        if diff < 0:
            warnings.warn("Opposing flow difference is negative; returning 0.0", RuntimeWarning)
            return 0.0
        return diff**(1.0 / n)
    else:
        raise ValueError("Flow type '{}' not recognized. Use 'assisting', 'opposing', or 'transverse'.".format(flow_type))


def mixed_convection_regime(Gr, Re):
    r'''Determines whether a convection process is predominantly forced,
    predominantly natural, or mixed (both significant) based on the
    Richardson parameter :math:`Gr / Re^2`, according to Section 9-6 of
    Çengel & Ghajar (5th Ed) [1]_.

    - :math:`Gr / Re^2 < 0.1`: Forced convection dominates (natural convection negligible).
    - :math:`Gr / Re^2 > 10`: Natural convection dominates (forced convection negligible).
    - :math:`0.1 \le Gr / Re^2 \le 10`: Mixed convection (both modes must be considered).

    Parameters
    ----------
    Gr : float
        Grashof number [-]
    Re : float
        Reynolds number [-]

    Returns
    -------
    regime : str
        'forced', 'natural', or 'mixed'

    Examples
    --------
    Example 9-7 from [1]_:
    >>> mixed_convection_regime(2.003e7, 4975.0)
    'mixed'

    References
    ----------
    .. [1] Çengel, Yunus A., and Afshin J. Ghajar. Heat and Mass Transfer:
       Fundamentals and Applications. 5th ed. New York: McGraw-Hill, 2015.
       Section 9-6, p. 563.
    '''
    if Re <= 0:
        return 'natural'
    param = Gr / (Re**2)
    if param < 0.1:
        return 'forced'
    elif param > 10.0:
        return 'natural'
    else:
        return 'mixed'


def h_air_natural_vertical_plate(delta_T, L):
    r'''Calculates simplified heat transfer coefficient for natural convection
    from a vertical plate in air at atmospheric pressure and moderate temperatures (~20-25 deg C).

    .. math::
        h = 1.42 (\Delta T / L)^{1/4} \quad \text{for laminar flow} \\
        h = 1.31 \Delta T^{1/3} \quad \text{for turbulent flow}

    Parameters
    ----------
    delta_T : float
        Temperature difference :math:`|T_s - T_\infty|`, [K]
    L : float
        Vertical plate height, [m]

    Returns
    -------
    h : float
        Heat transfer coefficient, [W/(m^2*K)]
    '''
    if delta_T <= 0 or L <= 0:
        return 0.0
    h_lam = 1.42 * (delta_T / L)**0.25
    h_turb = 1.31 * (delta_T**(1.0/3.0))
    return max(h_lam, h_turb)


def h_air_natural_horizontal_plate(delta_T, L, hot_surface_facing='up', is_hot_plate=True):
    r'''Calculates simplified heat transfer coefficient for natural convection
    from a horizontal plate in air at atmospheric pressure and moderate temperatures (~20-25 deg C).

    Parameters
    ----------
    delta_T : float
        Temperature difference :math:`|T_s - T_\infty|`, [K]
    L : float
        Characteristic length :math:`L_c = A_s / p`, [m]
    hot_surface_facing : str, optional
        'up' or 'down' [-]
    is_hot_plate : bool, optional
        True if hot, False if cold [-]

    Returns
    -------
    h : float
        Heat transfer coefficient, [W/(m^2*K)]
    '''
    if delta_T <= 0 or L <= 0:
        return 0.0
    assisted = (is_hot_plate and hot_surface_facing.lower() == 'up') or (not is_hot_plate and hot_surface_facing.lower() == 'down')
    if assisted:
        h_lam = 1.32 * (delta_T / L)**0.25
        h_turb = 1.52 * (delta_T**(1.0/3.0))
        return max(h_lam, h_turb)
    else:
        return 0.59 * (delta_T / L)**0.25


def h_air_natural_horizontal_cylinder(delta_T, D):
    r'''Calculates simplified heat transfer coefficient for natural convection
    from a horizontal cylinder in air at atmospheric pressure and moderate temperatures (~20-25 deg C).

    Parameters
    ----------
    delta_T : float
        Temperature difference :math:`|T_s - T_\infty|`, [K]
    D : float
        Outer diameter of cylinder, [m]

    Returns
    -------
    h : float
        Heat transfer coefficient, [W/(m^2*K)]
    '''
    if delta_T <= 0 or D <= 0:
        return 0.0
    h_lam = 1.32 * (delta_T / D)**0.25
    h_turb = 1.24 * (delta_T**(1.0/3.0))
    return max(h_lam, h_turb)
