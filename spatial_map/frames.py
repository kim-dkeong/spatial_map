import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def build_frame_table(frames, trials, cue_pos, frame_times, a, b,
                      n_img_frames, speed_smooth_s=0.25, teleport_thresh=1.0):

    ft = np.asarray(frame_times[:n_img_frames], float)
    t_vr = a * frames.time.to_numpy() + b                # json → vrec 
    z = frames.position_z.to_numpy()
    x = frames.position_x.to_numpy()

    # ---- nearest VR ----
    j = np.clip(np.searchsorted(t_vr, ft), 1, len(t_vr) - 1)
    j = np.where(np.abs(ft - t_vr[j - 1]) <= np.abs(t_vr[j] - ft), j - 1, j)
    inside = (ft >= t_vr[0]) & (ft <= t_vr[-1])

    step = np.r_[0, np.hypot(np.diff(x), np.diff(z))]
    tele = step > teleport_thresh
    dt = np.median(np.diff(t_vr))
    sp = gaussian_filter1d(np.where(tele, 0, step) / dt, sigma=speed_smooth_s / dt)

    F = pd.DataFrame(dict(
        t=ft, z=z[j], x=x[j], iState=frames.iState.to_numpy()[j],
        speed=np.interp(ft, t_vr, sp), inside=inside))

    tele_t = t_vr[tele]
    F["trial"] = -1
    F["cue"] = np.nan
    F["z_rel"] = np.nan
    for _, t in trials.iterrows():
        t0 = tele_t[np.searchsorted(tele_t, a * t.t_start + b - 0.05)]
        nxt = tele_t[tele_t > t0 + 0.1]
        t1 = nxt[0] if len(nxt) else t_vr[-1]
        m = (F.t >= t0) & (F.t < t1)
        F.loc[m, "trial"] = int(t.iTrial)
        F.loc[m, "cue"] = int(t.cue)
        cz = float(cue_pos.loc[cue_pos.iTrial == t.iTrial, "cue_z"].iloc[0])
        F.loc[m, "z_rel"] = F.loc[m, "z"] - cz           # cue = 0
    return F
