"""
행동(behavior) JSON, voltage recording(.mat), suite2p(.mat) 등 raw 데이터를
읽어서 표준화된 형태(DataFrame / dict)로 반환하는 함수들.
"""

import json

import numpy as np
import pandas as pd
import scipy.io as sio

FRAME_COLS = [
    "time", "frame", "iTrial", "iState", "iCue", "iChoice",
    "iCorrect", "iReward", "position_x", "position_z", "event",
]


def parse_behavior_json(json_path):
    """
    반환
      frames : 매 VR 프레임(60 Hz) 의 상태.  FRAME_COLS 컬럼.
               iTrial/iState/iCue/iChoice/iCorrect/iReward 는 '가장 최근 이벤트 값'
               으로 carry-forward 된 값입니다 (원래 쓰시던 방식 그대로).
      trials : trial 단위 요약 표. carry-forward 된 값이 아니라 **각 이벤트에서
               직접 읽은** 값이라 이쪽이 분석용으로 맞습니다.

    ※ 왜 두 개로 나누는가
      iChoice / iCorrect / iReward 는 **누적 카운터**입니다.
      iReward 는 15→30→45…, iCorrect 는 정답 누적 개수로 쌓입니다.
      그래서 carry-forward 하면 trial k 의 시작 구간에는 trial k-1 의 값이 붙습니다.
      (예: trial 4 의 start 레코드는 iChoice=2 인데 이건 trial 3 의 선택입니다.)
      조건별 분석에는 반드시 trials 표를 쓰세요.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    state = dict(iTrial=None, iState=None, iCue=None, iChoice=None,
                 iCorrect=None, iReward=None)
    rows, events = [], []

    for it in data:
        for k in state:
            if k in it:
                state[k] = it[k]

        # trial 이벤트 레코드 (note 가 있는 것) 를 따로 모아둔다
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

    # ---- trial 단위 요약 ----
    tr = []
    for k, g in ev.groupby("iTrial"):
        g = g.sort_values("time")
        st = g[g.note == "start"]
        cu = g[g.note == "cue_pr"]
        ch = g[g.note.isin(["left_pr", "right_pr"])]
        if len(st) == 0 or len(ch) == 0:
            continue                                     # 미완료 trial (마지막)
        tr.append(dict(
            iTrial=int(k),
            t_start=float(st.time.iloc[0]),
            t_cue=float(cu.time.iloc[0]) if len(cu) else np.nan,
            n_cue_events=len(cu),                        # 되돌아가서 재트리거된 횟수
            t_choice=float(ch.time.iloc[0]),
            cue=int(st.iCue.iloc[0]),                    # 1=left, 2=right
            choice=int(ch.iChoice.iloc[0]),              # 1=left, 2=right
            correct=int(ch.iCorrect.iloc[0]) - int(st.iCorrect.iloc[0]),   # 누적 → 차분
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
    iscell = np.asarray(s["iscell"])[:, 0] > 0.5         # 0열=분류, 1열=확률
    sig = np.asarray(s[use], float)[iscell]
    return dict(sig=sig, spks=np.asarray(s["spks"], float)[iscell],
                cell_idx=np.flatnonzero(iscell), n_roi=len(iscell),
                fs=float(s["ops"]["fs"]))
