import numpy as np


def sync_json_to_vrec(trials, vrec_trial, fit_slope=False, verbose=True):

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
            print("  ! 최대 오차가 imaging 1 frame(67 ms)을 넘습니다 — ")
    return a, b, resid
