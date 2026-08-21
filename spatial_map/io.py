import json

import numpy as np
import pandas as pd
import scipy.io as sio

FRAME_COLS = [
    "time", "frame", "iTrial", "iState", "iCue", "iChoice",
    "iCorrect", "iReward", "position_x", "position_z", "event",
]


def parse_behavior_json(json_path):
   
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    state = dict(iTrial=None, iState=None, iCue=None, iChoice=None,
                 iCorrect=None, iReward=None)
    rows, events = [], []

    for it in data:
        for k in state:
            if k in it:
                state[k] = it[k]

        # trial event
        if "note" in it and "iTrial" in it:
            events.append(dict(time=it["timeSecs"], note=it["note"],
                               **{k: it.get(k) for k in state}))

        if "position" in it:
            ev = it.get("events") or []
            rows.append(dict(time=it.get("timeSecs"), frame=it.get("frame"),
                             position_x=it["position"].get("x"),
                             position_z=it["position"].get("z"),
                             event=ev[0] if ev else None, **state))

    frames = pd.DataFrame(rows)[FRAME_COLS].sort_values("time").reset_index(drop=True)
    ev = pd.DataFrame(events)

    # ---- summary per trial ----
    tr = []
    for k, g in ev.groupby("iTrial"):
        g = g.sort_values("time")
        st = g[g.note == "start"]
        cu = g[g.note == "cue_pr"]
        ch = g[g.note.isin(["left_pr", "right_pr"])]
        if len(st) == 0 or len(ch) == 0:
            continue                                     
        tr.append(dict(
            iTrial=int(k),
            t_start=float(st.time.iloc[0]),
            t_cue=float(cu.time.iloc[0]) if len(cu) else np.nan,
            n_cue_events=len(cu),                        
            t_choice=float(ch.time.iloc[0]),
            cue=int(st.iCue.iloc[0]),                    # 1=left, 2=right
            choice=int(ch.iChoice.iloc[0]),              # 1=left, 2=right
            correct=int(ch.iCorrect.iloc[0]) - int(st.iCorrect.iloc[0]),   
            reward=int(ch.iReward.iloc[0]) - int(st.iReward.iloc[0]),
        ))
    trials = pd.DataFrame(tr).sort_values("iTrial").reset_index(drop=True)
    return frames, trials


def load_voltage_recording(path):
    v = sio.loadmat(path, simplify_cells=True)["vrec"]
    ch = {str(c).strip(): np.atleast_2d(np.asarray(d, float))
          for c, d in zip(v["channels"], v["data"])}
    trial = pd.DataFrame({k: np.atleast_1d(np.asarray(x)).ravel()
                          for k, x in v["trial"].items() if k != "nTrial"})
    return dict(
        channels=ch,
        frame_times=ch["AI 1"][:, 0],                    # imaging frame trigger onset
        vr_on=float(ch["AI 2"][0, 0]), vr_off=float(ch["AI 2"][0, 1]),
        reward=ch["AI 6"][:, 0] if ch["AI 6"].size else np.array([]),
        lick=ch["AI 7"][:, 0] if ch["AI 7"].size else np.array([]),
        trial=trial)


def load_suite2p(path, use="dff"):
    s = sio.loadmat(path, simplify_cells=True)["suite2p"]
    iscell = np.asarray(s["iscell"])[:, 0] > 0.5         
    sig = np.asarray(s[use], float)[iscell]
    return dict(sig=sig, spks=np.asarray(s["spks"], float)[iscell],
                cell_idx=np.flatnonzero(iscell), n_roi=len(iscell),
                fs=float(s["ops"]["fs"]))
