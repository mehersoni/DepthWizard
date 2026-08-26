"""
M4 Shadow Cue Module - Phase 4 Step 7 & Step 8: Building Height & Sensitivity / Uncertainty Analysis

Provides:
- pixel_to_physical_shadow_length(): Converts pixel length to physical meters given scale.
- compute_building_height(): Computes building height H = L_m * tan(radians(sun_elevation_deg)).
- estimate_building_height(): Complete interface with input validation and test/production status.
- compute_height_sensitivity(): Parametric sensitivity analysis for pixel length, scale, and solar elevation.
- propagate_height_uncertainty(): Analytical uncertainty propagation via partial derivatives.
"""

import math
from typing import Dict, Any, Optional, List


def pixel_to_physical_shadow_length(
    shadow_length_px: float,
    meters_per_pixel: Optional[float]
) -> Dict[str, Any]:
    """
    Convert geometric pixel shadow length to physical shadow length in meters.
    """
    if shadow_length_px is None or shadow_length_px <= 0.0:
        return {
            "shadow_length_px": float(shadow_length_px) if shadow_length_px is not None else 0.0,
            "meters_per_pixel": float(meters_per_pixel) if meters_per_pixel is not None else None,
            "physical_shadow_length_m": None,
            "is_calibrated": False,
            "status": "FAILED — Invalid pixel shadow length"
        }

    if meters_per_pixel is None or meters_per_pixel <= 0.0:
        return {
            "shadow_length_px": float(shadow_length_px),
            "meters_per_pixel": None,
            "physical_shadow_length_m": None,
            "is_calibrated": False,
            "status": "PHYSICAL LENGTH UNAVAILABLE — SCALE NOT CALIBRATED"
        }

    m_px = float(meters_per_pixel)
    l_m = float(shadow_length_px * m_px)

    return {
        "shadow_length_px": float(shadow_length_px),
        "meters_per_pixel": m_px,
        "physical_shadow_length_m": l_m,
        "is_calibrated": True,
        "status": "CALIBRATED"
    }


def compute_building_height(
    physical_shadow_length_m: Optional[float],
    sun_elevation_deg: Optional[float]
) -> Dict[str, Any]:
    """
    Compute estimated building height H = L_m * tan(radians(sun_elevation_deg)).
    """
    if physical_shadow_length_m is None or physical_shadow_length_m <= 0.0:
        return {
            "physical_shadow_length_m": None,
            "sun_elevation_deg": float(sun_elevation_deg) if sun_elevation_deg is not None else None,
            "building_height_m": None,
            "is_valid": False,
            "status": "HEIGHT UNAVAILABLE — Physical shadow length missing or uncalibrated"
        }

    if sun_elevation_deg is None:
        return {
            "physical_shadow_length_m": float(physical_shadow_length_m),
            "sun_elevation_deg": None,
            "building_height_m": None,
            "is_valid": False,
            "status": "HEIGHT UNAVAILABLE — Solar elevation angle missing (UNKNOWN)"
        }

    elev = float(sun_elevation_deg)
    if elev <= 0.0 or elev >= 90.0:
        return {
            "physical_shadow_length_m": float(physical_shadow_length_m),
            "sun_elevation_deg": elev,
            "building_height_m": None,
            "is_valid": False,
            "status": f"FAILED — Invalid solar elevation angle ({elev}°). Must be in range (0, 90)."
        }

    tan_theta = math.tan(math.radians(elev))
    height_m = float(physical_shadow_length_m * tan_theta)

    return {
        "physical_shadow_length_m": float(physical_shadow_length_m),
        "sun_elevation_deg": elev,
        "building_height_m": height_m,
        "is_valid": True,
        "status": "COMPUTED"
    }


def estimate_building_height(
    shadow_length_px: Optional[float],
    meters_per_pixel: Optional[float],
    sun_elevation_deg: Optional[float],
    pair_confidence: float = 1.0,
    is_test_mode: bool = True
) -> Dict[str, Any]:
    """
    Complete Building Height Estimation Interface with input validation and test/production status.
    """
    if shadow_length_px is None or not isinstance(shadow_length_px, (int, float)) or shadow_length_px <= 0.0:
        return {
            "shadow_length_px": float(shadow_length_px) if shadow_length_px is not None else None,
            "meters_per_pixel": float(meters_per_pixel) if meters_per_pixel is not None else None,
            "shadow_length_m": None,
            "sun_elevation_deg": float(sun_elevation_deg) if sun_elevation_deg is not None else None,
            "height_m": None,
            "status": "[HEIGHT UNAVAILABLE]",
            "reason": "INVALID_SHADOW_LENGTH",
            "pair_confidence": float(pair_confidence),
            "height_confidence_status": "[HEIGHT UNAVAILABLE]"
        }

    if meters_per_pixel is None or not isinstance(meters_per_pixel, (int, float)):
        return {
            "shadow_length_px": float(shadow_length_px),
            "meters_per_pixel": None,
            "shadow_length_m": None,
            "sun_elevation_deg": float(sun_elevation_deg) if sun_elevation_deg is not None else None,
            "height_m": None,
            "status": "[HEIGHT UNAVAILABLE]",
            "reason": "SCALE_UNAVAILABLE",
            "pair_confidence": float(pair_confidence),
            "height_confidence_status": "[HEIGHT UNAVAILABLE]"
        }

    if meters_per_pixel <= 0.0:
        return {
            "shadow_length_px": float(shadow_length_px),
            "meters_per_pixel": float(meters_per_pixel),
            "shadow_length_m": None,
            "sun_elevation_deg": float(sun_elevation_deg) if sun_elevation_deg is not None else None,
            "height_m": None,
            "status": "[HEIGHT UNAVAILABLE]",
            "reason": "INVALID_SCALE",
            "pair_confidence": float(pair_confidence),
            "height_confidence_status": "[HEIGHT UNAVAILABLE]"
        }

    if sun_elevation_deg is None or not isinstance(sun_elevation_deg, (int, float)):
        return {
            "shadow_length_px": float(shadow_length_px),
            "meters_per_pixel": float(meters_per_pixel),
            "shadow_length_m": float(shadow_length_px * meters_per_pixel),
            "sun_elevation_deg": None,
            "height_m": None,
            "status": "[HEIGHT UNAVAILABLE]",
            "reason": "SOLAR_ELEVATION_UNAVAILABLE",
            "pair_confidence": float(pair_confidence),
            "height_confidence_status": "[HEIGHT UNAVAILABLE]"
        }

    if sun_elevation_deg <= 0.0 or sun_elevation_deg >= 90.0:
        return {
            "shadow_length_px": float(shadow_length_px),
            "meters_per_pixel": float(meters_per_pixel),
            "shadow_length_m": float(shadow_length_px * meters_per_pixel),
            "sun_elevation_deg": float(sun_elevation_deg),
            "height_m": None,
            "status": "[HEIGHT UNAVAILABLE]",
            "reason": "INVALID_SOLAR_ELEVATION",
            "pair_confidence": float(pair_confidence),
            "height_confidence_status": "[HEIGHT UNAVAILABLE]"
        }

    l_m = float(shadow_length_px * meters_per_pixel)
    tan_theta = math.tan(math.radians(sun_elevation_deg))
    height_m = float(l_m * tan_theta)

    if is_test_mode:
        status_label = "[TEST ONLY]"
        conf_status = "[TEST HEIGHT]"
    else:
        status_label = "[PRODUCTION HEIGHT]"
        conf_status = "[PRODUCTION HEIGHT]"

    return {
        "shadow_length_px": float(shadow_length_px),
        "meters_per_pixel": float(meters_per_pixel),
        "shadow_length_m": l_m,
        "sun_elevation_deg": float(sun_elevation_deg),
        "height_m": height_m,
        "status": status_label,
        "reason": "COMPUTED_SUCCESSFULLY",
        "pair_confidence": float(pair_confidence),
        "height_confidence_status": conf_status
    }


def propagate_height_uncertainty(
    shadow_length_px: float,
    meters_per_pixel: float,
    sun_elevation_deg: float,
    delta_L_px: float = 1.0,
    delta_scale: float = 0.05,
    delta_sun_deg: float = 2.0
) -> Dict[str, Any]:
    """
    Step 8.6 Analytical Uncertainty Propagation for H = L * s * tan(theta).

    Partial derivatives:
    - dH/dL = s * tan(theta)
    - dH/ds = L * tan(theta)
    - dH/dtheta = L * s * sec^2(theta_rad)
    """
    L = float(shadow_length_px)
    s = float(meters_per_pixel)
    theta_deg = float(sun_elevation_deg)
    theta_rad = math.radians(theta_deg)

    tan_th = math.tan(theta_rad)
    cos_th = math.cos(theta_rad)
    sec2_th = 1.0 / (cos_th * cos_th)

    H_nominal = L * s * tan_th

    # Partial derivatives
    dH_dL = s * tan_th
    dH_ds = L * tan_th
    dH_dtheta_rad = L * s * sec2_th

    # Convert angular uncertainty delta_sun_deg to radians
    delta_theta_rad = math.radians(delta_sun_deg)

    # Variance components
    var_L = (dH_dL * delta_L_px) ** 2
    var_s = (dH_ds * delta_scale) ** 2
    var_theta = (dH_dtheta_rad * delta_theta_rad) ** 2

    total_std_m = float(math.sqrt(var_L + var_s + var_theta))

    return {
        "nominal_height_m": H_nominal,
        "total_std_m": total_std_m,
        "var_component_L_m": float(math.sqrt(var_L)),
        "var_component_s_m": float(math.sqrt(var_s)),
        "var_component_theta_m": float(math.sqrt(var_theta)),
        "partial_dH_dL": dH_dL,
        "partial_dH_ds": dH_ds,
        "partial_dH_dtheta": dH_dtheta_rad
    }
