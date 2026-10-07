"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
"""

import warnings

import pytest
from fluids.numerics import assert_close, assert_close1d

from ht import (
    Bier,
    Cooper,
    Forster_Zuber,
    Gorenflo,
    HEDH_Montinsky,
    HEDH_Taborek,
    McNelly,
    Montinsky,
    Rohsenow,
    Serth_HEDH,
    Stephan_Abdelsalam,
    Zuber,
    h_nucleic,
    h_nucleic_methods,
    qmax_boiling,
    qmax_boiling_methods,
)

### Nucleic boiling


def test_boiling_nucleic_Rohsenow():
    # Checked with 10.30 Problem set 8.
    h_calc = [Rohsenow(Te=i, Cpl=4180, kl=0.688, mul=2.75E-4, sigma=0.0588, Hvap=2.25E6, rhol=958, rhog=0.597, Csf=0.013, n=1) for i in [4.3, 9.1, 13]]
    h_values = [2860.6242230238613, 12811.697777642301, 26146.321995188344]
    assert_close1d(h_calc, h_values)
    q_test = Rohsenow(Te=4.9, Cpl=4217., kl=0.680, mul=2.79E-4, sigma=0.0589, Hvap=2.257E6, rhol=957.854, rhog=0.595593, Csf=0.011, n=1.26)*4.9
    assert_close(18245.91080863059, q_test)

    h_Te = 1316.2269561541964
    h_q = Rohsenow(q=5*h_Te, Cpl=4180, kl=0.688, mul=2.75E-4, sigma=0.0588, Hvap=2.25E6, rhol=958, rhog=0.597)
    assert_close(h_Te, h_q)

    with pytest.raises(Exception):
        Rohsenow(Cpl=4180, kl=0.688, mul=2.75E-4, sigma=0.0588, Hvap=2.25E6, rhol=958, rhog=0.597)


def test_boiling_nucleic_McNelly():
    # Water matches expectations, ammonia is somewhat distant. Likely just
    # error in the text's calculation.
    h_McNelly1 = McNelly(Te=4.3, P=101325, Cpl=4180., kl=0.688, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    h_McNelly2 = McNelly(Te=9.1, P=101325, Cpl=4472., kl=0.502, sigma=0.0325, Hvap=1.37E6, rhol=689., rhog=0.843)
    assert_close1d([h_McNelly1, h_McNelly2], [533.8056972951352, 6387.3951029225855])
    # Check the the solution with q gives the same h
    h_Te = 533.8056972951352
    h_q= McNelly(q=4.3*h_Te, P=101325, Cpl=4180., kl=0.688, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    assert_close(h_Te, h_q)
    with pytest.raises(Exception):
        McNelly(P=101325, Cpl=4472., kl=0.502, sigma=0.0325, Hvap=1.37E6, rhol=689., rhog=0.843)


def test_boiling_nucleic_Forster_Zuber():
    # All examples are for water from [1]_ and match.
    # 4th example is from [3]_ and matches completely.
    FZ1 = Forster_Zuber(Te=4.3, dPsat=3906*4.3, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    FZ2 = Forster_Zuber(Te=9.1, dPsat=3906*9.1, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    FZ3 = Forster_Zuber(Te=13, dPsat=3906*13, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    FZ4 = Forster_Zuber(Te=16.2, dPsat=106300., Cpl=2730., kl=0.086, mul=156E-6, sigma=.0082, Hvap=272E3, rhol=567., rhog=18.09)
    FZ_values = [3519.9239897462644, 7393.507072909551, 10524.54751261952, 5512.279068294656]
    assert_close1d([FZ1, FZ2, FZ3, FZ4], FZ_values)
    h_Te = 3519.9239897462644
    h_q = Forster_Zuber(q=4.3*h_Te, dPsat=3906*4.3, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    assert_close(h_Te, h_q)
    with pytest.raises(Exception):
         Forster_Zuber(dPsat=3906*4.3, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)


def test_boiling_nucleic_Montinsky():
    # Fourth example is from [4]_ and matches to within the error of the algebraic
    # manipulation rounding.
    # First three examples are for water, ammonia, and benzene, from [1]_, and
    # match to within 20%.
    W_Te = [Montinsky(Te=i, P=101325., Pc=22048321.0) for i in [4.3, 9.1, 13]]
    W_Te_values = [1185.0509770292663, 6814.079848742471, 15661.924462897328]
    assert_close1d(W_Te, W_Te_values)
    A_Te = [Montinsky(Te=i, P=101325., Pc=112E5) for i in [4.3, 9.1, 13]]
    A_Te_values = [377.04493949460635, 2168.0200886557072, 4983.118427770712]
    assert_close1d(A_Te, A_Te_values)
    B_Te = [Montinsky(Te=i, P=101325., Pc=48.9E5) for i in [4.3, 9.1, 13]]
    B_Te_values = [96.75040954887533, 556.3178536987874, 1278.6771501657056]
    assert_close1d(B_Te, B_Te_values)
    assert_close(Montinsky(310.3E3, 2550E3, 16.2), 2423.2656339862583)

    h_Te = 1185.0509770292663
    h_q = Montinsky(q=4.3*h_Te, P=101325., Pc=22048321.0)
    assert_close(h_Te, h_q)
    with pytest.raises(Exception):
        Montinsky(P=101325., Pc=22048321.0)


def test_boiling_nucleic_Stephan_Abdelsalam():
    # Stephan Abdelsalam function; allow bad function method
    Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,  sigma=0.0082, Hvap=272E3, rhol=567.0, rhog=18.09, angle=35.0, correlation="fail")

    cs = ["general", "water", "hydrocarbon", "cryogenic", "refrigerant"]
    h_SA = [Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, correlation=i) for i in cs]
    # Propane-like properties in every variant, so the water value is not
    # physical; it only checks the formula. X3 = Cpl*Tsat*d^2/alpha^2 here is
    # 4.4 times X4 = Hvap*d^2/alpha^2.
    h_values = [26722.441071108373, 9147145.25760636, 21009.03422203015, 15460.943960447508, 84657.98595551957]
    assert_close1d(h_SA, h_values)

    h_qs = []
    for h, c in zip(h_values, cs):
        h_qs.append(Stephan_Abdelsalam(q=16.2*h, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567.0, rhog=18.09, correlation=c))
    assert_close1d(h_qs, h_values)

    with pytest.raises(Exception):
        Stephan_Abdelsalam(Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,  sigma=0.0082, Hvap=272E3, rhol=567.0, rhog=18.09)

    # Water at 100 C, q = 100 kW/m^2: the water variant agrees with the
    # general variant to within 5% (it was 66% higher when X3 duplicated X4)
    w = dict(rhol=958.35, rhog=0.5976, mul=2.817e-4, kl=0.6791, Cpl=4215.7, Hvap=2.2565e6, sigma=0.05891)
    h_water = Stephan_Abdelsalam(Tsat=373.15, q=1E5, correlation="water", **w)
    h_general = Stephan_Abdelsalam(Tsat=373.15, q=1E5, correlation="general", **w)
    assert_close(h_water, 8888.501043845223)
    assert_close(h_water, h_general, rtol=0.06)

    # A supplied angle overrides the automatic one (45 degrees for water)
    h_45 = Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, correlation="water", angle=45.0)
    h_35 = Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, correlation="water", angle=35.0)
    assert_close(h_45, h_values[1])
    assert_close(h_35, 19232960.16736815)


def test_boiling_nucleic_HEDH_Taborek():
    h = HEDH_Taborek(Te=16.2, P=310.3E3, Pc=2550E3)
    assert_close(h, 1397.272486525486)

    h_q = HEDH_Taborek(P=310.3E3, Pc=2550E3, q=16.2*1397.272486525486)
    assert_close(h, h_q)

    with pytest.raises(Exception):
        HEDH_Taborek(P=310.3E3, Pc=2550E3)


def test_boiling_nucleic_Bier():
    h_W = [Bier(Te=i, P=101325., Pc=22048321.0) for i in [4.3, 9.1, 13]]
    h_W_values = [1290.5349471503353, 7420.6159464293305, 17056.026492351128]
    assert_close1d(h_W, h_W_values)
    h_B = [Bier(101325., 48.9E5, i) for i in [4.3, 9.1, 13]]
    h_B_values = [77.81190344679615, 447.42085661013226, 1028.3812069865799]
    assert_close1d(h_B, h_B_values)

    h_Te = 1290.5349471503353
    h_q = Bier(101325., 22048321.0, q=4.3*h_Te)
    assert_close(h_Te, h_q)

    with pytest.raises(Exception):
        Bier(P=310.3E3, Pc=2550E3)


def test_boiling_nucleic_Cooper():
    h_W = [Cooper(Te=i, P=101325., Pc=22048321.0, MW=18.02) for i in [4.3, 9.1, 13]]
    h_W_values = [1558.1435442153575, 7138.700876530947, 14727.09551225091]
    assert_close1d(h_W, h_W_values)
    h_B = [Cooper(101325., 48.9E5, 78.11184, i) for i in [4.3, 9.1, 13]]
    h_B_values = [504.57942247904055, 2311.7520711767947, 4769.130145905329]
    assert_close1d(h_B, h_B_values)

    h_Te = 1558.1435442153575
    h_q = Cooper(P=101325., Pc=22048321.0, MW=18.02, q=h_Te*4.3)
    assert_close(h_Te, h_q)

    with pytest.raises(Exception):
        Cooper(P=101325., Pc=22048321.0, MW=18.02)


def test_Gorenflo():
    # Water at 3 bar uses the water-specific F(p*) and n
    q = 2E4
    h1 = Gorenflo(P=3E5, Pc=22048320., q=q, CASRN="7732-18-5")
    Pr = 3E5/22048320.
    Fp = 1.73*Pr**0.27 + (6.1 + 0.68/(1 - Pr))*Pr**2
    assert_close(h1, 5600.0*Fp)
    assert_close(h1, 3043.344595525422)
    h2 = Gorenflo(P=3E5, Pc=22048320., Te=q/h1, CASRN="7732-18-5")
    assert_close(h1, h2)

    # Ethanol at 3 bar; dPdT and sigma do not affect a tabulated fluid
    h1 = Gorenflo(P=3E5, Pc=6137000., q=q, CASRN="64-17-5")
    assert_close(h1, 2828.6050440749627)
    assert_close(h1, Gorenflo(3E5, 6137000., dPdT=3800., sigma=0.016, q=q, CASRN="64-17-5"))
    h2 = Gorenflo(P=3E5, Pc=6137000., Te=q/h1, CASRN="64-17-5")
    assert_close(h1, h2)

    # Custom h0, no CASRN
    h = Gorenflo(3E5, 6137000., q=2E4, h0=3700.0)
    Pr = 3E5/6137000.
    assert_close(h, 3700.0*(0.7*Pr**0.2 + 4.0*Pr + 1.4*Pr/(1.0 - Pr)))

    # R134a, copper wall
    h = Gorenflo(1E6, 4059280., q=2E4, CASRN='811-97-2')
    assert_close(h, 8282.243199918714)
    h = Gorenflo(1E6, 4059280., Te=2.0, CASRN='811-97-2')
    assert_close(h, 4663.248093617287)
    assert_close(Gorenflo(1E6, 4059280., q=2.0*h, CASRN='811-97-2'), h)

    # Stainless-steel wall reduces the result by (7730/35350)^0.5
    h_ss = Gorenflo(1E6, 4059280., q=2E4, CASRN='811-97-2', eff=7730.)
    assert_close(h_ss, 3872.960046980562)

    # At the reference state (p* = 0.1, q0, Ra0, copper) h ~= tabulated h0
    assert_close(Gorenflo(0.1*4059280., 4059280., q=2E4, CASRN='811-97-2'), 4200.0, rtol=5e-3)
    assert_close(Gorenflo(0.1*22048320., 22048320., q=2E4, CASRN='7732-18-5'), 5600.0, rtol=5e-3)

    # Helium h0 is tabulated at q0 = 1 kW/m^2 (Table H2.1 footnote i)
    assert_close(Gorenflo(0.1*227000., 227000., q=1E3, CASRN='7440-59-7'), 2000.0, rtol=5e-3)

    with pytest.raises(Exception):
        # CAS number not in the database
        Gorenflo(3E5, 6137000., q=2E4, CASRN="6400-17-5")
    with pytest.raises(Exception):
        # Neither Te nor q provided
        Gorenflo(3E5, 6137000., CASRN="64-17-5")
    with pytest.raises(ValueError):
        # No way to obtain h0
        Gorenflo(3E5, 6137000., q=2E4)
    with pytest.raises(ValueError):
        # Supercritical
        Gorenflo(7E6, 6137000., q=2E4, CASRN="64-17-5")


def test_Gorenflo_h0_estimate():
    """Eq. (8): h0 = 3580*P_f^0.6 for fluids not in Table H2.1."""
    # R134a properties at p* = 0.1 from Table H2.1: P_f = 1.333, alpha_0,calc = 4.26 kW/m^2/K
    Pc = 4059000.
    h = Gorenflo(0.1*Pc, Pc, dPdT=13630., sigma=10.226e-3, q=2E4)
    assert_close(h, 4241.803323721339)
    Pr = 0.1
    Fp = 0.7*Pr**0.2 + 4.0*Pr + 1.4*Pr/(1.0 - Pr)
    assert_close(h/Fp, 4260.0, rtol=2e-3)

    # Reference fluid: P_f = 1 by definition -> h0 = 3580
    from ht.boiling_nucleic import gorenflo_fluid_aliases, h0_VDI_2e
    assert_close(h0_VDI_2e["ReferenceFluid"], 3580.0)
    assert gorenflo_fluid_aliases["ReferenceFluid"] == "reference"
    Pr = 1e5/1e6
    Fp = 0.7*Pr**0.2 + 4.0*Pr + 1.4*Pr/(1.0 - Pr)
    h = Gorenflo(P=1e5, Pc=1e6, q=2e4, fluid='ReferenceFluid')
    assert_close(h, 3580.0*Fp)
    assert_close(h, Gorenflo(P=1e5, Pc=1e6, q=2e4, fluid='Reference'))
    # ...or with P_f supplied, the Eq. (8) estimate
    h = Gorenflo(P=1e5, Pc=1e6, dPdT=2e4, sigma=0.01, q=2e4, fluid='ReferenceFluid')
    assert_close(h, 3580.0*2.0**0.6*Fp)

    # A tabulated fluid ignores dPdT and sigma
    assert_close(Gorenflo(1E6, 4059280., dPdT=1., sigma=1., q=2E4, CASRN='811-97-2'),
                 Gorenflo(1E6, 4059280., q=2E4, CASRN='811-97-2'))


def test_Gorenflo_Te_warning():
    """Specifying Te warns that input errors are amplified by 1/(1-n)."""
    # Water at 1 atm: n = 0.766, so 1/(1-n) = 4.3 and 10% -> about 50%
    with pytest.warns(UserWarning, match=r"1/\(1-n\) = 4\.3.*about a 50% error"):
        Gorenflo(101325., 22090e3, Te=15.0, CASRN="7732-18-5")
    # No Te warning when q is given
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Gorenflo(101325., 22090e3, q=2E4, CASRN="7732-18-5")


def test_Gorenflo_footnote_warnings():
    """Table H2.1 footnotes d/e (data quality) warn but do not change h."""
    from ht.boiling_nucleic import _gorenflo_footnote_d, _gorenflo_footnote_e

    for casrn in ("92-52-4", "71-36-3", "78-83-1", "754-12-1", "75-73-0",
                  "306-83-2", "7782-44-7", "7440-37-1", "7440-01-9", "1333-74-0"):
        assert casrn in _gorenflo_footnote_d
    for casrn in ("71-43-2", "108-88-3", "56-23-5"):
        assert casrn in _gorenflo_footnote_e

    with pytest.warns(UserWarning, match="footnote d"):
        h = Gorenflo(P=2E5, Pc=5050000., q=2E4, CASRN="7782-44-7")
    Pr = 2E5/5050000.
    assert_close(h, 9500.0*(0.7*Pr**0.2 + 4.0*Pr + 1.4*Pr/(1.0 - Pr)))

    with pytest.warns(UserWarning, match="footnote e"):
        Gorenflo(P=2E5, Pc=4894000., q=2E4, CASRN="71-43-2")

    # Wall-material correction applies to cryogens too
    with pytest.warns(UserWarning):
        h_ss = Gorenflo(P=2E5, Pc=5050000., q=2E4, CASRN="7782-44-7", eff=7730.)
    assert_close(h_ss/h, (7730./35350.)**0.5)

    # No warning for unflagged fluids, or when h0 is supplied directly
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Gorenflo(1E6, 4059280., q=2E4, CASRN='811-97-2')
        Gorenflo(3E5, 22048320., q=2E4, CASRN="7732-18-5")
        Gorenflo(P=2E5, Pc=5050000., q=2E4, CASRN="7782-44-7", h0=9500.)


def test_h_nucleic():
    h = h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Method="Rohsenow")
    assert_close(h, 1094.0242011089285)

    h = h_nucleic(Te=4.3, P=101325.0, Cpl=4180., kl=0.688, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597, Method="McNelly")
    assert_close(h, 533.8056972951352)

    h = h_nucleic(Te=4.3, dPsat=3906*4.3, Cpl=4180., kl=0.688, mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597, Method="Forster-Zuber")
    assert_close(h, 3519.9239897462644)

    h = h_nucleic(P=101325, Pc=22048321, Te=4.3, Method="Montinsky")
    assert_close(h, 1185.0509770292663)

    h = h_nucleic(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, Method="Stephan-Abdelsalam")
    assert_close(h, 26722.441071108373)

    h = h_nucleic(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, Method="Stephan-Abdelsalam water", CAS="7732-18-5")
    assert_close(h, 9147145.25760636)

    h = h_nucleic(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, Method="Stephan-Abdelsalam cryogenic", CAS="1333-74-0")
    assert_close(h, 15460.943960447508)

    h = h_nucleic(Te=16.2, P=310.3E3, Pc=2550E3, Method="HEDH-Taborek")
    assert_close(h, 1397.272486525486)

    h = h_nucleic(P=101325., Pc=22048321.0, Te=4.3, Method="Bier")
    assert_close(h, 1290.5349471503353)

    h = h_nucleic(P=101325., Pc=22048321.0, MW=18.02, Te=4.3, Method="Cooper")
    assert_close(h, 1558.1435442153575)

    # Gorenflo (2010) via h_nucleic — water at 3 bar
    h = h_nucleic(P=3E5, Pc=22048320., q=2E4, CAS="7732-18-5", Method="Gorenflo (2010)")
    assert_close(h, 3043.344595525422)

    # Test the kwargs
    h = h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Method="Rohsenow", Csf=0.011, n=1.26)
    assert_close(h, 3723.655267067467)


    # methods (Gorenflo listed when the CAS has a tabulated h0)
    methods = h_nucleic_methods(P=101325., Pc=22048321.0, MW=18.02, dPsat=3906*4.3, dPdT=3906., Tsat=437.5, CAS="7732-18-5", rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9)
    assert len(methods) == 10

    methods = h_nucleic_methods(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6, sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, CAS="1333-74-0")
    assert len(methods) == 3

    with pytest.raises(Exception):
        h_nucleic(P=101325., Pc=22048321.0, Te=4.3, Method="BADMETHOD")

    with pytest.raises(Exception):
        h_nucleic()

    # test the default method
    h = h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9)
    assert_close(h, 1094.0242011089285)


def test_qmax_Zuber():
    q_calc_ex = Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567.0, rhog=18.09, K=0.149)
    assert_close(q_calc_ex, 444307.22304342285)
    q_max = Zuber(8.2E-3, 272E3, 567, 18.09, 0.18)
    assert_close(q_max, 536746.9808578263)


def test_qmax_Serth_HEDH():
    qmax = Serth_HEDH(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567.0, rhog=18.09)
    assert_close(qmax, 351867.46522901946)
    # Test K calculated as a function of R
    qmax = Serth_HEDH(0.00127, 8.2E-3, 272E3, 567, 18.09)
    assert_close(qmax, 440111.4740326096)


def test_HEDH_Montinsky():
    assert_close(HEDH_Montinsky(310.3E3, 2550E3), 398405.66545181436)


def test_qmax_nucleic():
    q = qmax_boiling(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    assert_close(q, 351867.46522901946)

    q = qmax_boiling(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, Method="Zuber")
    assert_close(q, 536746.9808578263)

    q = qmax_boiling(P=310.3E3, Pc=2550E3)
    assert_close(q, 398405.66545181436)

    with pytest.raises(Exception):
        qmax_boiling(D=0.0127)
    with pytest.raises(Exception):
        qmax_boiling(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, Method="BADMETHOD")


    methods = qmax_boiling_methods(P=310.3E3, Pc=2550E3, D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    assert len(methods) == 3
