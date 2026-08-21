"""
spatial_map
===========

VR 미로 행동 로그 + imaging 데이터를 정렬하고, dF/F 를 공간(z) bin 으로
묶어 spatial map / raster-PSTH 를 만드는 도구 모음.

사용 예
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

# 개인적으로 쓰시는 로컬 패키지(ephys)에 대한 의존성입니다.
# pip 로 설치되지 않는 사설 모듈이라, 이 패키지를 다른 환경(다른 PC, CI 등)에
# 설치할 때 없을 수 있습니다. 없어도 나머지 함수들은 정상 동작하도록
# 실패해도 무시하고 넘어가게 해두었습니다. 실제로 필요하시면
# 이 저장소와 별도로 ephys 패키지를 pip install -e 로 설치해두세요.
try:
    from ephys.ophys import Ophys, VRec, Suite2p  # noqa: F401
except ImportError:
    Ophys = VRec = Suite2p = None

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
