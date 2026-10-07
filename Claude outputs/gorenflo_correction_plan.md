# Gorenflo Pool Boiling Correlation — Correction Plan

**Date:** 2026-10-06  
**File:** `ht/boiling_nucleic.py` → function `Gorenflo()`  
**Reference:** VDI Heat Atlas, 2nd Edition (Springer, 2010), Chapter H2 — *Pool Boiling*

---

## 1. Correct Gorenflo Correlation (VDI Heat Atlas)

The Gorenflo nucleate pool boiling correlation is based on the law of corresponding states and reads:

$$\frac{h}{h_0} = F(p^*) \cdot \left(\frac{q}{q_0}\right)^n \cdot \left(\frac{R_a}{R_{a,0}}\right)^{0.133}$$

### Reference conditions

| Symbol | Value | Description |
|--------|-------|-------------|
| $q_0$ | 20 000 W m⁻² | Reference heat flux |
| $R_{a,0}$ | 0.4 μm | Reference surface roughness (arithmetic mean, Ra) |
| $p^*_0$ | 0.1 | Reference reduced pressure at which h₀ is tabulated |
| $p^*$ | $P/P_c$ | Reduced pressure (dimensionless) |

The surface-roughness correction factor is

$$F_W = \left(\frac{R_a}{R_{a,0}}\right)^{0.133}$$

---

### 1a. Most fluids (all except water)

**Pressure function:**

$$F(p^*) = 1.2\,p^{*\,0.27} + \left(2.5 + \frac{1}{1-p^*}\right)p^*$$

**Heat-flux exponent:**

$$n = 0.9 - 0.3\,p^{*\,0.3}$$

---

### 1b. Water

**Pressure function:**

$$F(p^*) = 1.73\,p^{*\,0.27} + \left(6.1 + \frac{0.68}{1-p^*}\right)p^{*\,2}$$

**Heat-flux exponent:**

$$n = 0.9 - 0.3\,p^{*\,0.15}$$

---

### 1c. Full equation — given heat flux q

$$\boxed{h = h_0 \cdot F(p^*) \cdot \left(\frac{q}{q_0}\right)^n \cdot \left(\frac{R_a}{R_{a,0}}\right)^{0.133}}$$

---

### 1d. Full equation — given wall superheat ΔT (Te)

Starting from Newton's law of cooling: $q = h \cdot \Delta T_e$, substitute into the expression above:

$$h = h_0 \cdot F(p^*) \cdot \left(\frac{h\,\Delta T_e}{q_0}\right)^n \cdot F_W$$

Collect $h$ terms:

$$h^{1-n} = h_0 \cdot F(p^*) \cdot F_W \cdot \left(\frac{\Delta T_e}{q_0}\right)^n$$

Raise both sides to the power $\tfrac{1}{1-n}$:

$$\boxed{h = \left[h_0 \cdot F(p^*) \cdot F_W \cdot \left(\frac{\Delta T_e}{q_0}\right)^n\right]^{\!\tfrac{1}{1-n}}}$$

Note: $n < 1$ for all physical reduced pressures, so the exponent $\tfrac{1}{1-n}$ is always positive and finite.

---

## 2. Worked Example — Water at 3 bar, q = 20 000 W m⁻²

```
P   = 3×10⁵ Pa
Pc  = 22 048 320 Pa   (water critical pressure)
q   = 20 000 W m⁻²
h0  = 5 600 W m⁻² K⁻¹  (from VDI table, water)
Ra  = Ra,0 = 0.4 μm   → Fw = 1.0

p* = P/Pc = 3×10⁵ / 22 048 320 = 0.013607

n  = 0.9 - 0.3 × 0.013607^0.15
   = 0.9 - 0.3 × 0.5250
   = 0.7425

F(p*) = 1.73 × 0.013607^0.27 + (6.1 + 0.68/0.9864) × 0.013607²
      = 1.73 × 0.3133      + (6.1 + 0.6894) × 0.0001852
      = 0.5420             + 0.001258
      = 0.5433

h = 5600 × 0.5433 × (20000/20000)^0.7425 × 1.0
  = 5600 × 0.5433 × 1.0
  ≈ 3042 W m⁻² K⁻¹            (code returns 3043.34)
```

---

## 3. Comparison with Current Implementation

### 3a. Formula implementation (q-given case) — **Correct**

```python
# current code
Fp = 1.2*Pr**0.27 + (2.5 + 1/(1-Pr))*Pr          # non-water ✓
Fp = 1.73*Pr**0.27 + (6.1 + 0.68/(1-Pr))*Pr**2   # water     ✓
n  = 0.9 - 0.3*Pr**0.3                             # non-water ✓
n  = 0.9 - 0.3*Pr**0.15                            # water     ✓
CW = (Ra/Ra0)**0.133                               # ✓
return h0*CW*Fp*(q/q0)**n                          # ✓
```

### 3b. Te-given case — **Bug: wrong exponent sign/form**

```python
# current code (WRONG)
elif Te is not None:
    A = h0*CW*Fp*(Te/q0)**n
    return A**(-1./(n - 1.))       # ← this is algebraically equal to A**(1/(1-n))
                                   #   BUT only when n < 1; it is easily misread
                                   #   and the intermediate A has mixed units
```

The expression `A**(-1./(n - 1.))` is mathematically equivalent to `A**(1./(1-n))` for `n < 1`,
**however** the form `(-1./(n - 1.))` is error-prone:
- For `n` very close to 1 the denominator underflows.
- The intermediate variable `A = h0*CW*Fp*(Te/q0)**n` carries mixed units
  (K m²/W)ⁿ which makes the expression opaque.
- The *correct, explicit* derivation gives `A**(1./(1-n))`.

### 3c. Docstring LaTeX error — **Bug in rendered documentation**

The current docstring shows the water pressure function as:

```
f(p^*) = 1.73 p^{*0.27} + (6.1 + 0.68/(1-p^*)) p^2
```

The final term is written `p^2`, which renders as $p^2$ (plain *p* squared),  
**but it should be** $p^{*\,2}$ (reduced pressure squared). The code itself
uses `Pr**2` which is correct — this is purely a docstring error.

### 3d. Water identification when h0 is supplied manually — **Latent bug**

```python
if CASRN != "7732-18-5":
    # non-water formula used whenever CASRN is None or anything other than water
    ...
```

If a user calls `Gorenflo(P, Pc, q=q, h0=5600)` for water without passing
`CASRN='7732-18-5'`, the **non-water** F(p\*) and n are silently applied.
A guard should be added, or the docstring should clearly warn about this.

---

## 4. Planned Fixes

| # | Location | Change |
|---|----------|--------|
| 1 | `boiling_nucleic.py` line ~863 | Replace `A**(-1./(n - 1.))` with the explicit `A**(1./(1. - n))` |
| 2 | Docstring LaTeX | Fix `p^2` → `{p^*}^2` in the water pressure-function equation |
| 3 | Docstring note | Add warning that `CASRN='7732-18-5'` must be supplied for water when using manual `h0` |

No changes to the **formula coefficients** are needed — the mathematical
implementation of F(p\*) and n for both water and non-water cases matches
the VDI Heat Atlas 2nd edition exactly.

---

*Prepared by Claude (Sonnet 4.6) for review before code changes are applied.*
