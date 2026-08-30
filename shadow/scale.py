"""
M4 Shadow Cue Module - Phase 4 Step 4: Physical Scale Calibration & Management

Provides a centralized interface for physical image scale (meters_per_pixel).
Prevents hard-coding scale values and enforces explicit validation before physical conversions.
"""

from typing import Dict, Any, Optional


class PhysicalScaleManager:
    """
    Centralized scale manager for converting pixel distances to physical ground distances (meters).
    """

    def __init__(self, meters_per_pixel: Optional[float] = None, source_description: str = "Uncalibrated"):
        self.set_scale(meters_per_pixel, source_description)

    def set_scale(self, meters_per_pixel: Optional[float], source_description: str = "Manual Configuration"):
        """Set or update physical scale parameter (meters_per_pixel)."""
        if meters_per_pixel is not None:
            if meters_per_pixel <= 0.0:
                raise ValueError(f"Invalid meters_per_pixel ({meters_per_pixel}). Must be strictly positive.")
            self.meters_per_pixel = float(meters_per_pixel)
            self.source_description = source_description
            self.is_calibrated = True
            if "Manual" in source_description or "Test" in source_description:
                self.confidence_status = "[MODERATE CONFIDENCE]"
            else:
                self.confidence_status = "[HIGH CONFIDENCE]"
        else:
            self.meters_per_pixel = None
            self.source_description = "No metadata or physical reference available"
            self.is_calibrated = False
            self.confidence_status = "[NOT AVAILABLE]"

    def convert_pixels_to_meters(self, pixel_distance: float) -> Optional[float]:
        """Convert pixel length to physical meters if scale is calibrated."""
        if not self.is_calibrated or self.meters_per_pixel is None:
            return None
        return float(pixel_distance * self.meters_per_pixel)

    def get_status_report(self) -> Dict[str, Any]:
        """Return structured calibration diagnostic dictionary."""
        return {
            "is_calibrated": self.is_calibrated,
            "meters_per_pixel": self.meters_per_pixel,
            "source_description": self.source_description,
            "confidence_status": self.confidence_status,
            "status_text": "PHYSICAL SCALE VALIDATED" if self.is_calibrated else "PHYSICAL SCALE NOT VALIDATED"
        }
