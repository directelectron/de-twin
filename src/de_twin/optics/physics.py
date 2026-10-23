"""Electron-optical constants and small helper formulas."""

from __future__ import annotations

import math

ELECTRON_CHARGE_C = 1.602176634e-19
REST_ENERGY_KEV = 510.99895


def electron_wavelength_nm(ht_kv: float) -> float:
    """Relativistic electron wavelength (the C++ ``ElectronWavelengthNm``, kV in)."""
    v = (ht_kv if ht_kv > 0 else 200.0) * 1000.0
    return 1.226426 / math.sqrt(v * (1.0 + 0.9784760e-6 * v))


def interaction_constant(ht_kv: float) -> float:
    """sigma in rad / (V nm): phase shift per volt of projected potential per nm."""
    kv = ht_kv if ht_kv > 0 else 200.0
    lam = electron_wavelength_nm(kv)
    return (2.0 * math.pi / (lam * kv * 1000.0)) * (REST_ENERGY_KEV + kv) / (2.0 * REST_ENERGY_KEV + kv)


def focal_spread_nm(
    cc_mm: float,
    energy_spread_ev_fwhm: float,
    ht_kv: float,
    current_instability: float = 0.0,
    ht_instability: float = 0.0,
) -> float:
    """1-sigma defocus spread Delta from chromatic aberration.

    Delta = Cc * sqrt((sigma_E/E*)^2 + (2 sigma_I/I)^2 + (sigma_V/V)^2) with the
    relativistic correction E* (Reimer). FWHM energy spread -> sigma via 2.355.
    """
    kv = ht_kv if ht_kv > 0 else 200.0
    eps = kv / REST_ENERGY_KEV
    rel = (1.0 + eps) / (1.0 + eps / 2.0)
    s_e = (energy_spread_ev_fwhm / 2.3548) / (kv * 1000.0) * rel
    return cc_mm * 1e6 * math.sqrt(s_e**2 + (2.0 * current_instability) ** 2 + ht_instability**2)
