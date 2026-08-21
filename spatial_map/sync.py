"""json 시계와 voltage recording 시계를 맞추는 함수."""

import numpy as np


def sync_json_to_vrec(trials, vrec_trial, fit_slope=False, verbose=True):
    """
    json 시계 → voltage recording 시계.   t_vrec = a * t_json + b

    필요한 건 **offset 한 개**입니다. 두 시계 모두에서 보이는 같은 이벤트
    (trial start / cue / choice) 를 짝지어 차이의 중앙값을 취하면 끝입니다.

        offset = median(vrec 이벤트 시각 - json 이벤트 시각)

    fit_slope=True 로 기울기(clock drift)까지 fit 할 수 있지만, 이 데이터에서
    drift 는 -26 ppm = 세션 전체 250 s 에 걸쳐 6.5 ms 로 imaging 1 frame(67 ms)
    의 1/10 입니다. 기본값은 offset 만 (a=1).

    ※ offset 을 AI2 onset 으로 직접 잡지 마세요.
      AI2 는 "VR 이 도는 동안 high" 인 신호라 onset 이 json 의 어느 시각에
      해당하는지는 세팅에 따라 달라집니다. 이 데이터에서는 json t=0 이 아니라
      **첫 trial start(json 5.79 s)** 에 대응해서, json t=0 으로 가정하면
      전체가 5.79 s 밀립니다. trial 이벤트로 맞추면 그런 가정이 아예 필요 없습니다.
    """
    n = min(len(trials), len(vrec_trial))
    tj = np.concatenate([trials.t_start[:n], trials.t_cue[:n], trials.t_choice[:n]])
    tv = np.concatenate([vrec_trial['timeStart'][:n], vrec_trial['timeCue'][:n],
                         vrec_trial['timeChoice'][:n]])
    ok = np.isfinite(tj) & np.isfinite(tv)
    tj, tv = tj[ok], tv[ok]

    if fit_slope:
        a, b = np.polyfit(tj, tv, 1)
    else:
        a, b = 1.0, float(np.median(tv - tj))
    resid = tv - (a * tj + b)
    rms = float(np.sqrt((resid ** 2).mean()))

    if verbose:
        print(f"[sync] anchor {len(tj)}개  →  t_vrec = t_json {b:+.4f} s"
              + (f"  (slope {a:.9f})" if fit_slope else ""))
        print(f"[sync] 검증: RMS {rms*1000:.1f} ms, 최대 "
              f"{np.abs(resid).max()*1000:.1f} ms")
        if np.abs(resid).max() > 0.067:
            print("  ! 최대 오차가 imaging 1 frame(67 ms)을 넘습니다 — "
                  "이벤트 짝짓기나 trial 개수를 확인하세요")
    return a, b, resid
