"""
spatial_map
===========

-------
    from spatial_map import (
        parse_behavior_json, load_voltage_recording, load_suite2p,
        sync_json_to_vrec, find_cue_position, find_choice_position,
        build_frame_table, spatial_map, zscore, smooth, plot_raster_psth,
    )
"""

from .io import (
    FRAME_COLS,
    parse_behavior_json,
    load_voltage_recording,
    load_suite2p,
)
from .sync import sync_json_to_vrec
from .events import find_cue_position, find_choice_position
from .frames import build_frame_table
from .maps import spatial_map, zscore, smooth
from .viz import plot_raster_psth, COLOR, LABEL


__all__ = [
    "FRAME_COLS",
    "parse_behavior_json",
    "load_voltage_recording",
    "load_suite2p",
    "sync_json_to_vrec",
    "find_cue_position",
    "find_choice_position",
    "build_frame_table",
    "spatial_map",
    "zscore",
    "smooth",
    "plot_raster_psth",
    "COLOR",
    "LABEL",
    "Ophys",
    "VRec",
    "Suite2p",
]

__version__ = "0.1.0"
