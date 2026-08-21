"""VR 프레임 로그에서 trial 별 cue / choice 지점의 좌표를 뽑아내는 함수들."""

import numpy as np
import pandas as pd


def find_cue_position(frames, trials):
    """
    json frame 레코드의 events=['cue'] 마커로 trial 별 cue 발생 좌표를 구한다.
    cue 는 위치 트리거이므로 z 가 거의 일정해야 정상이다.
    """
    cues = frames[frames.event == "cue"]
    out = []
    for _, t in trials.iterrows():
        g = cues[(cues.time >= t.t_start) & (cues.time <= t.t_choice)]
        if len(g) == 0:
            out.append(dict(iTrial=t.iTrial, cue_z=np.nan, cue_x=np.nan,
                            cue_time=np.nan, n_retrigger=0))
            continue
        f = g.iloc[0]
        out.append(dict(iTrial=int(t.iTrial), cue_z=float(f.position_z),
                        cue_x=float(f.position_x), cue_time=float(f.time),
                        n_retrigger=len(g) - 1))
    return pd.DataFrame(out)


def find_choice_position(frames, trials):
    """events=['left']/['right'] 마커 → 선택(arm 진입) 좌표."""
    ch = frames[frames.event.isin(["left", "right"])]
    out = []
    for _, t in trials.iterrows():
        g = ch[(ch.time >= t.t_start) & (ch.time <= t.t_choice + 0.5)]
        f = g.iloc[0] if len(g) else None
        out.append(dict(iTrial=int(t.iTrial),
                        choice_z=float(f.position_z) if f is not None else np.nan,
                        choice_x=float(f.position_x) if f is not None else np.nan))
    return pd.DataFrame(out)
