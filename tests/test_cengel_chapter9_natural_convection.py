'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import math
import pytest
from fluids.numerics import assert_close

from ht import (
    # Immersed bodies & surfaces
    Nu_horizontal_cylinder_Churchill_Chu,
    Nu_vertical_plate_Churchill,
    Nu_vertical_plate_laminar_Cengel,
    Nu_vertical_plate_turbulent_Cengel,
    Nu_vertical_plate_Cengel,
    Nu_inclined_plate_Cengel,
    Nu_horizontal_plate_Cengel,
    is_vertical_cylinder_plate_like,
    Nu_vertical_plate_isoflux_Cengel,
    # Channels & fin arrays
    Ra_vertical_channel,
    Nu_vertical_channel_isothermal_Bar_Cohen,
    optimum_fin_spacing_isothermal,
    Nu_optimum_fin_spacing,
    Q_finned_surface_natural,
    Ra_star_vertical_channel,
    Nu_vertical_channel_isoflux_Bar_Cohen,
    optimum_spacing_isoflux,
    Q_PCBs_natural,
    # Mixed convection
    Nu_mixed_convection,
    mixed_convection_regime,
    # Simplified air relations
    h_air_natural_vertical_plate,
    h_air_natural_horizontal_plate,
    h_air_natural_horizontal_cylinder,
    # Enclosed cavities
    Q_enclosure_natural,
    k_eff_enclosure,
    Nu_horizontal_enclosure_Jakob,
    Nu_horizontal_enclosure_Globe_Dropkin,
    Nu_horizontal_enclosure_Hollands,
    Nu_inclined_enclosure_Hollands,
    critical_angle_inclined_enclosure,
    Nu_inclined_enclosure_Catton,
    Nu_inclined_enclosure_Arnold,
    Nu_vertical_enclosure_Berkovsky_Polevikov_1,
    Nu_vertical_enclosure_Berkovsky_Polevikov_2,
    Nu_vertical_enclosure_MacGregor_Emery_laminar,
    Nu_vertical_enclosure_MacGregor_Emery_turbulent,
    Nu_vertical_enclosure_Cengel,
    # Concentric geometries
    F_concentric_cylinders,
    k_eff_concentric_cylinders_Raithby_Hollands,
    Q_concentric_cylinders_natural,
    F_concentric_spheres,
    k_eff_concentric_spheres_Raithby_Hollands,
    Q_concentric_spheres_natural,
    # Radiation helper
    emissivity_effective_parallel_plates,
)


def test_cengel_example_9_1_pipe():
    """Çengel & Ghajar (5th Ed), Example 9-1: Heat Loss from Hot-Water Pipes
    Horizontal pipe D = 0.08 m, L = 6 m, Ts = 70 C, Tinf = 20 C.
    Properties at Tf = 45 C: k = 0.02699 W/m*K, Pr = 0.7241, nu = 1.750e-5 m^2/s, beta = 1/318 K^-1.
    Rayleigh number Ra_D = 1.867e6.
    Textbook: Nu = 17.39, h = 5.867 W/m^2*K, Q = 442 W.
    """
    D = 0.08
    L = 6.0
    Pr = 0.7241
    Ra = 1.867e6
    k = 0.02699
    Gr = Ra / Pr

    Nu = Nu_horizontal_cylinder_Churchill_Chu(Pr, Gr)
    assert_close(Nu, 17.39, rtol=1e-3)

    h = k * Nu / D
    assert_close(h, 5.867, rtol=1e-3)

    A_s = math.pi * D * L
    Q = h * A_s * (70.0 - 20.0)
    assert_close(Q, 442.0, rtol=5e-3)

    # Test simplified air relation
    h_simple = h_air_natural_horizontal_cylinder(delta_T=50.0, D=D)
    assert_close(h_simple, 1.32 * (50.0 / D)**0.25, rtol=1e-4)


def test_cengel_example_9_2_plate_orientations():
    """Çengel & Ghajar (5th Ed), Example 9-2: Cooling of a Plate in Different Orientations
    Plate 0.6 m x 0.6 m, Ts = 90 C, Tinf = 30 C.
    Properties at Tf = 60 C: k = 0.02808 W/m*K, Pr = 0.7202, nu = 1.896e-5 m^2/s, beta = 1/333 K^-1.
    (a) Vertical: L = 0.6 m, Ra = 7.649e8, Gr = 1.062066e9.
        Textbook: Nu_Churchill = 113.3, h = 5.302 W/m^2*K, Q = 115 W.
        Textbook: Nu_simple = 98.12 (Eq. 9-19).
    (b) Horizontal with hot surface facing up:
        Lc = As/p = 0.15 m, Ra = 1.195e7, Gr = 1.65926e7.
        Textbook: Nu = 31.75 (Eq. 9-22), h = 5.944 W/m^2*K, Q = 128 W.
    (c) Horizontal with hot surface facing down:
        Textbook: Nu = 15.87 (Eq. 9-24), h = 2.971 W/m^2*K, Q = 64.2 W.
    """
    Pr = 0.7202
    k = 0.02808
    delta_T = 60.0
    A_s = 0.6 * 0.6

    # (a) Vertical plate
    L = 0.6
    Ra_vert = 7.649e8
    Gr_vert = Ra_vert / Pr

    Nu_church = Nu_vertical_plate_Cengel(Pr, Gr_vert, method='Churchill')
    assert_close(Nu_church, 113.34, rtol=1e-3)
    h_church = k * Nu_church / L
    assert_close(h_church, 5.305, rtol=1e-3)
    Q_church = h_church * A_s * delta_T
    assert_close(Q_church, 114.6, rtol=1e-2)

    Nu_simple = Nu_vertical_plate_Cengel(Pr, Gr_vert, method='simple')
    assert_close(Nu_simple, 98.12, rtol=1e-3)
    assert_close(Nu_vertical_plate_laminar_Cengel(Ra_vert), 98.12, rtol=1e-3)

    # (b) Horizontal plate - hot surface facing up
    Lc = 0.15
    Ra_horiz = 1.195e7
    Gr_horiz = Ra_horiz / Pr

    Nu_up_laminar = Nu_horizontal_plate_Cengel(Pr, Gr_horiz, hot_surface_facing='up', is_hot_plate=True, flow_regime='laminar')
    assert_close(Nu_up_laminar, 31.75, rtol=1e-3)
    h_up = k * Nu_up_laminar / Lc
    assert_close(h_up, 5.944, rtol=1e-3)
    Q_up = h_up * A_s * delta_T
    assert_close(Q_up, 128.4, rtol=1e-2)

    # Default auto-selection at Ra = 1.195e7 (> 1e7 selects Eq. 9-23 turbulent)
    Nu_up_auto = Nu_horizontal_plate_Cengel(Pr, Gr_horiz, hot_surface_facing='up', is_hot_plate=True)
    assert_close(Nu_up_auto, 34.29, rtol=1e-3)

    # (c) Horizontal plate - hot surface facing down
    Nu_down = Nu_horizontal_plate_Cengel(Pr, Gr_horiz, hot_surface_facing='down', is_hot_plate=True)
    assert_close(Nu_down, 15.87, rtol=1e-3)
    h_down = k * Nu_down / Lc
    assert_close(h_down, 2.972, rtol=1e-3)
    Q_down = h_down * A_s * delta_T
    assert_close(Q_down, 64.2, rtol=1e-2)

    # Cold surface orientation symmetry
    Nu_cold_down = Nu_horizontal_plate_Cengel(Pr, Gr_horiz, hot_surface_facing='down', is_hot_plate=False, flow_regime='laminar')
    assert_close(Nu_cold_down, Nu_up_laminar, rtol=1e-5)
    Nu_cold_up = Nu_horizontal_plate_Cengel(Pr, Gr_horiz, hot_surface_facing='up', is_hot_plate=False)
    assert_close(Nu_cold_up, Nu_down, rtol=1e-5)

    # Simplified air relations
    h_air_v = h_air_natural_vertical_plate(delta_T=delta_T, L=L)
    assert_close(h_air_v, max(1.42 * (delta_T / L)**0.25, 1.31 * delta_T**(1.0/3.0)), rtol=1e-4)

    h_air_h_up = h_air_natural_horizontal_plate(delta_T=delta_T, L=Lc, hot_surface_facing='up', is_hot_plate=True)
    assert_close(h_air_h_up, max(1.32 * (delta_T / Lc)**0.25, 1.52 * delta_T**(1.0/3.0)), rtol=1e-4)

    h_air_h_down = h_air_natural_horizontal_plate(delta_T=delta_T, L=Lc, hot_surface_facing='down', is_hot_plate=True)
    assert_close(h_air_h_down, 0.59 * (delta_T / Lc)**0.25, rtol=1e-4)


def test_cengel_inclined_plate_and_vertical_cylinder():
    """Tests for inclined plate g*cos(theta) scaling and vertical cylinder plate-like criterion (Eq. 9-28)."""
    Pr = 0.7202
    Gr = 1.062066e9

    # At theta = 0, inclined plate must match vertical plate exactly
    Nu_vert = Nu_vertical_plate_Cengel(Pr, Gr, method='Churchill')
    Nu_inc_0 = Nu_inclined_plate_Cengel(Pr, Gr, theta=0.0, method='Churchill')
    assert_close(Nu_vert, Nu_inc_0, rtol=1e-7)

    # At theta = 45 deg, Gr_eff = Gr * cos(45 deg)
    Nu_inc_45 = Nu_inclined_plate_Cengel(Pr, Gr, theta=45.0, hot_surface_facing='down', is_hot_plate=True)
    Gr_eff = Gr * math.cos(math.radians(45.0))
    Nu_expected = Nu_vertical_plate_Cengel(Pr, Gr_eff, method='Churchill')
    assert_close(Nu_inc_45, Nu_expected, rtol=1e-7)

    # Vertical cylinder plate-like criterion: D >= 35 * L / Gr^(1/4) (Eq. 9-28)
    L = 0.20
    Gr_cyl = 1e8
    # D_crit = 35 * 0.20 / (1e8)^0.25 = 7.0 / 100 = 0.07 m
    assert is_vertical_cylinder_plate_like(L=L, D=0.10, Gr=Gr_cyl) is True
    assert is_vertical_cylinder_plate_like(L=L, D=0.05, Gr=Gr_cyl) is False


def test_cengel_example_9_3_heat_sink_fins():
    """Çengel & Ghajar (5th Ed), Example 9-3: Optimum Fin Spacing of a Heat Sink
    Heat sink W = 0.12 m, L = 0.18 m, Ts = 80 C, Tinf = 30 C, fin thickness t = 1 mm.
    Fin height from base H = 2.4 cm = 0.024 m.
    Properties at Tf = 55 C: k = 0.02772 W/m*K, Pr = 0.7215, nu = 1.847e-5 m^2/s, beta = 1/328 K^-1.
    Ra_L = 1.845e7.
    Textbook: S_opt = 7.45 mm (0.00745 m), Nu_opt = 1.307, h = 4.863 W/m^2*K,
              n_fins = 14, Q = 29.4 W.
    """
    L = 0.18
    W = 0.12
    t = 0.001
    H = 0.024  # Fin height from base (2.4 cm)
    Ra_L = 1.845e7
    k = 0.02772
    Ts = 80.0
    Tinf = 30.0

    S_opt = optimum_fin_spacing_isothermal(L, Ra_L)
    assert_close(S_opt, 0.007454, rtol=1e-3)

    Nu_opt = Nu_optimum_fin_spacing()
    assert_close(Nu_opt, 1.307, rtol=1e-5)

    h = Nu_opt * k / S_opt
    assert_close(h, 4.861, rtol=1e-3)

    n_fins = W / (S_opt + t)
    assert_close(n_fins, 14.19, rtol=1e-2)

    # Heat dissipation for 14 fins of length L = 0.18 m and protrusion height H = 0.024 m
    Q = Q_finned_surface_natural(h=h, n=14, L=L, H=H, Ts=Ts, Tinf=Tinf)
    assert_close(Q, 29.40, rtol=5e-3)

    # Test isothermal vertical channel correlation (Bar-Cohen & Rohsenow, Eq. 9-31)
    Gr_S = (Ra_L / 0.7215) * (S_opt / L)**3
    Ra_S = Ra_vertical_channel(Pr=0.7215, Gr=Gr_S, S=S_opt, L=L)
    Nu_ch = Nu_vertical_channel_isothermal_Bar_Cohen(Ra_S, S=S_opt, L=L)
    assert_close(Nu_ch, 1.307, rtol=5e-2)


def test_cengel_vertical_channel_isoflux_pcbs():
    """Tests for vertical channel with uniform heat flux (PCBs, Eqs. 9-35 to 9-38)."""
    k = 0.026
    nu = 1.6e-5
    Pr = 0.71
    L = 0.20
    g = 9.81
    beta = 1.0 / 300.0
    q_s = 100.0  # W/m^2
    S = 0.015  # 15 mm spacing

    Ra_star = Ra_star_vertical_channel(Pr=Pr, g=g, beta=beta, qs=q_s, S=S, k=k, nu=nu)
    assert Ra_star > 0

    Nu_ch_flux = Nu_vertical_channel_isoflux_Bar_Cohen(Ra_star, S=S, L=L)
    assert Nu_ch_flux > 0

    S_opt_flux = optimum_spacing_isoflux(k=k, nu=nu, L=L, g=g, beta=beta, qs=q_s, Pr=Pr)
    assert 0.001 < S_opt_flux < 0.05

    # Total heat dissipation for n PCBs (Eq. 9-38)
    Q_pcb = Q_PCBs_natural(qs=q_s, n=10, L=L, H=0.15)
    assert_close(Q_pcb, 2.0 * 10 * (q_s * L * 0.15), rtol=1e-5)


def test_cengel_example_9_4_double_pane_window():
    """Çengel & Ghajar (5th Ed), Example 9-4: Heat Loss through a Double-Pane Window
    Vertical window H = 0.8 m, W = 2.0 m, gap L = 0.02 m. Aspect ratio H/L = 40.
    T1 = 12 C, T2 = 2 C, Tavg = 7 C = 280 K.
    Properties: k = 0.02416 W/m*K, Pr = 0.7344, Ra_L = 1.050e4.
    Textbook: Nu = 1.40, k_eff = 0.03385 W/m*K (using 0.02416*1.40 = 0.03382 W/m*K),
              Q = 27.1 W, epsilon_eff = 0.818 (for eps1=eps2=0.9).
    """
    H = 0.8
    W = 2.0
    L = 0.02
    aspect_ratio = H / L
    Pr = 0.7344
    Ra_L = 1.050e4
    k = 0.02416
    T1 = 12.0
    T2 = 2.0
    A_s = H * W

    # MacGregor & Emery laminar correlation (Eq. 9-54)
    Nu = Nu_vertical_enclosure_MacGregor_Emery_laminar(Pr, Ra_L, aspect_ratio)
    assert_close(Nu, 1.401, rtol=1e-3)

    # Master vertical enclosure selector
    Nu_master = Nu_vertical_enclosure_Cengel(Pr, Ra_L, aspect_ratio)
    assert_close(Nu_master, Nu, rtol=1e-6)

    # Effective thermal conductivity
    k_eff = k_eff_enclosure(k, Nu)
    assert_close(k_eff, 0.03385, rtol=2e-3)

    # Heat transfer rate
    Q = Q_enclosure_natural(k_eff, A_s, T1, T2, L)
    assert_close(Q, 27.08, rtol=1e-2)

    # Effective emissivity for parallel glass plates
    eps_eff = emissivity_effective_parallel_plates(0.9, 0.9)
    assert_close(eps_eff, 0.8182, rtol=1e-3)


def test_cengel_example_9_5_concentric_spheres():
    """Çengel & Ghajar (5th Ed), Example 9-5: Pipe Insulation for Thermal Burn Prevention
    Concentric spheres: Di = 3.0 m, Do = 3.15 m, Lc = (Do - Di)/2 = 0.075 m.
    Properties at Tavg = 95 C: k = 0.03060 W/m*K, Pr = 0.7122, Ra_L = 1.7342e6.
    Textbook: F_sph = 0.0007602, k_eff = 0.1119 W/m*K.
    """
    Di = 3.0
    Do = 3.15
    Pr = 0.7122
    Ra = 1.7342e6
    k = 0.03060
    Ti = 150.0
    To = 40.0

    F_sph = F_concentric_spheres(Di, Do)
    assert_close(F_sph, 0.0007602, rtol=1e-3)

    k_eff = k_eff_concentric_spheres_Raithby_Hollands(Pr, Ra, Di, Do, k)
    assert_close(k_eff, 0.1119, rtol=1e-3)

    # Heat rate between concentric spheres (Eq. 9-59)
    Lc = (Do - Di) / 2.0
    Q = Q_concentric_spheres_natural(k_eff, Di, Do, Ti, To)
    Q_expected = k_eff * math.pi * (Di * Do / Lc) * (Ti - To)
    assert_close(Q, Q_expected, rtol=1e-6)


def test_cengel_concentric_cylinders():
    """Tests for concentric horizontal cylinders (Eqs. 9-56, 9-57, 9-58)."""
    Di = 0.10
    Do = 0.20
    Pr = 0.71
    Ra = 1e5
    k = 0.026

    F_cyl = F_concentric_cylinders(Di, Do)
    assert F_cyl > 0

    k_eff = k_eff_concentric_cylinders_Raithby_Hollands(Pr, Ra, Di, Do, k)
    assert k_eff > k

    Q_per_m = Q_concentric_cylinders_natural(k_eff, Di, Do, Ti=100.0, To=20.0)
    assert_close(Q_per_m, (2.0 * math.pi * k_eff / math.log(Do / Di)) * (100.0 - 20.0), rtol=1e-6)

    # Conduction limit when F_cyl * Ra < 100
    k_eff_low = k_eff_concentric_cylinders_Raithby_Hollands(Pr, Ra=1.0, Di=Di, Do=Do, k=k)
    assert_close(k_eff_low, k, rtol=1e-6)


def test_cengel_horizontal_and_inclined_enclosures():
    """Tests for horizontal and inclined enclosures (Jakob, Globe-Dropkin, Hollands, Catton, Arnold)."""
    # Horizontal enclosure: Hollands (Eq. 9-47)
    # Pure conduction for Ra < 1708
    assert_close(Nu_horizontal_enclosure_Hollands(1000.0), 1.0, rtol=1e-6)
    # Convection active for Ra = 1e5
    Nu_h_1e5 = Nu_horizontal_enclosure_Hollands(1e5)
    assert_close(Nu_h_1e5, 3.994, rtol=1e-3)

    # Horizontal enclosure: Jakob (Eqs. 9-44, 9-45)
    assert_close(Nu_horizontal_enclosure_Jakob(1000.0), 1.0, rtol=1e-6)
    assert_close(Nu_horizontal_enclosure_Jakob(1e5, regime='laminar'), 0.195 * (1e5)**0.25, rtol=1e-5)
    assert_close(Nu_horizontal_enclosure_Jakob(1e6, regime='turbulent'), 0.068 * (1e6)**(1.0/3.0), rtol=1e-5)

    # Horizontal enclosure: Globe-Dropkin (Eq. 9-46)
    Nu_gd = Nu_horizontal_enclosure_Globe_Dropkin(Pr=5.0, Ra=1e6)
    assert_close(Nu_gd, 0.069 * (1e6)**(1.0/3.0) * (5.0)**0.074, rtol=1e-5)

    # Inclined enclosure: Critical angles (Table 9-2)
    assert_close(critical_angle_inclined_enclosure(1.0), 25.0, rtol=1e-5)
    assert_close(critical_angle_inclined_enclosure(3.0), 53.0, rtol=1e-5)
    assert_close(critical_angle_inclined_enclosure(12.0), 70.0, rtol=1e-5)

    # Inclined enclosure: Hollands (Eq. 9-48)
    Nu_inc_h = Nu_inclined_enclosure_Hollands(Ra=5e4, theta=30.0, H=0.6, L=0.05)
    assert 1.0 < Nu_inc_h < 10.0

    # Inclined enclosure: Catton (Eqs. 9-49, 9-50)
    Nu_catton = Nu_inclined_enclosure_Catton(Ra=1e5, theta=45.0, aspect_ratio=5.0, Pr=0.71)
    assert Nu_catton > 1.0

    # Inclined enclosure: Arnold (Eq. 9-51 for theta > 90 deg)
    Nu_arnold = Nu_inclined_enclosure_Arnold(Nu_vertical=2.5, theta=120.0)
    assert_close(Nu_arnold, 1.0 + (2.5 - 1.0) * math.sin(math.radians(120.0)), rtol=1e-5)


def test_cengel_vertical_enclosure_correlations():
    """Tests for Berkovsky-Polevikov and MacGregor-Emery turbulent relations."""
    Pr = 0.71
    Ra = 1e5

    # Berkovsky-Polevikov 1 (H/L <= 2, Eq. 9-52)
    Nu_bp1 = Nu_vertical_enclosure_Berkovsky_Polevikov_1(Pr, Ra)
    assert Nu_bp1 > 1.0

    # Berkovsky-Polevikov 2 (2 < H/L <= 10, Eq. 9-53)
    Nu_bp2 = Nu_vertical_enclosure_Berkovsky_Polevikov_2(Pr, Ra, aspect_ratio=5.0)
    assert Nu_bp2 > 1.0

    # MacGregor-Emery turbulent (Ra > 1e6, Eq. 9-55)
    Nu_me_turb = Nu_vertical_enclosure_MacGregor_Emery_turbulent(Ra=1e7)
    assert_close(Nu_me_turb, 0.046 * (1e7)**(1.0/3.0), rtol=1e-5)


def test_cengel_example_9_7_mixed_convection():
    """Çengel & Ghajar (5th Ed), Example 9-7: Mixed Convection during Heating of Milk
    Solar collector pipe D = 0.05 m, L = 6 m.
    Re = 4975, Gr = 2.003e7.
    Textbook: Gr/Re^2 = 0.809 (0.1 < Gr/Re^2 < 10 -> Mixed convection).
              Pure forced: Nu_f = 42.14, Pure natural: Nu_n = 36.46.
              Assisting flow (n=3): Nu = 49.8.
              Opposing flow (n=3): Nu = 29.8.
    """
    Re = 4975.0
    Gr = 2.003e7
    Nu_f = 42.14
    Nu_n = 36.46

    # Regime identification
    regime = mixed_convection_regime(Gr, Re)
    assert regime == 'mixed'

    # Assisting flow
    Nu_assist = Nu_mixed_convection(Nu_f, Nu_n, flow_type='assisting', n=3.0)
    assert_close(Nu_assist, 49.77, rtol=1e-3)

    # Opposing flow
    Nu_oppose = Nu_mixed_convection(Nu_f, Nu_n, flow_type='opposing', n=3.0)
    assert_close(Nu_oppose, 29.77, rtol=1e-3)

    # Transverse flow
    Nu_trans = Nu_mixed_convection(Nu_f, Nu_n, flow_type='transverse', n=3.0)
    assert_close(Nu_trans, Nu_assist, rtol=1e-6)

    # Pure forced and natural regimes
    assert mixed_convection_regime(Gr=100.0, Re=10000.0) == 'forced'
    assert mixed_convection_regime(Gr=1e8, Re=100.0) == 'natural'


def test_warnings_and_exceptions():
    """Tests that out-of-bounds inputs raise appropriate warnings and invalid inputs raise exceptions."""
    # Invalid method in Nu_vertical_plate_Cengel
    with pytest.raises(ValueError):
        Nu_vertical_plate_Cengel(0.71, 1e6, method='unknown')

    # Invalid flow_type in Nu_mixed_convection
    with pytest.raises(ValueError):
        Nu_mixed_convection(10.0, 10.0, flow_type='invalid')

    # Out of range warning for laminar vertical plate
    with pytest.warns(RuntimeWarning):
        Nu_vertical_plate_laminar_Cengel(1e11)

    # Out of range warning for turbulent vertical plate
    with pytest.warns(RuntimeWarning):
        Nu_vertical_plate_turbulent_Cengel(1e6)

    # Out of range warning for inclined plate tilt angle > 60 deg
    with pytest.warns(RuntimeWarning):
        Nu_inclined_plate_Cengel(0.71, 1e6, theta=75.0)

    # Invalid geometry for concentric cylinders (Do <= Di)
    with pytest.raises(ValueError):
        F_concentric_cylinders(Di=0.2, Do=0.1)

    # Invalid geometry for concentric spheres (Do <= Di)
    with pytest.raises(ValueError):
        F_concentric_spheres(Di=0.5, Do=0.4)
