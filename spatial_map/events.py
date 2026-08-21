import numpy as np
import pandas as pd


def find_cue_position(frames, trials):
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
    ch = frames[frames.event.isin(["left", "right"])]
    out = []
    for _, t in trials.iterrows():
        g = ch[(ch.time >= t.t_start) & (ch.time <= t.t_choice + 0.5)]
        f = g.iloc[0] if len(g) else None
        out.append(dict(iTrial=int(t.iTrial),
                        choice_z=float(f.position_z) if f is not None else np.nan,
                        choice_x=float(f.position_x) if f is not None else np.nan))
    return pd.DataFrame(out)
