"""Processing (M3): plain float64 XRF reductions on JAX (native device policy)."""
from .backend import backend_info, current_device, require_backend
from .reductions import (
    calibrate,
    channel_range_for_energy,
    energy_to_channel,
    map_from_channels,
    spectrum_from_points,
    spectrum_from_roi,
)

__all__ = [
    "backend_info",
    "calibrate",
    "channel_range_for_energy",
    "current_device",
    "energy_to_channel",
    "map_from_channels",
    "require_backend",
    "spectrum_from_points",
    "spectrum_from_roi",
]