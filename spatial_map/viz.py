import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

# maze range: -3~9
COLOR = {1: "#2f6fd0", 2: "#c0392b"}          # 1=left, 2=right
LABEL = {1: "Left", 2: "Right"}
_INK, _MUTED = "#1a1a1a", "#6b6b6b"


def plot_raster_psth(maps, centers, trial_cue, cell, ax=None, raster="heatmap",
                     landmarks=(0.0, 4.0), title=None, cbar=True,
                     order=(2, 1), save=None, xlabel="z − cue z (VR unit)"):

    maps = np.asarray(maps)
    trial_cue = np.asarray(trial_cue)
    if maps.shape[1] != len(trial_cue):
        raise ValueError(f"trial 수가 안 맞습니다: maps {maps.shape[1]}, "
                         f"trial_cue {len(trial_cue)}. spatial_map 이 반환한 "
                         f"ids 순서로 trial_cue 를 만들었는지 확인하세요.")
    R = maps[cell]                                        # (n_trials, n_bins)

    

    idx = {c: np.flatnonzero(trial_cue == c) for c in order}
    row_order = np.concatenate([idx[c] for c in reversed(order)])  
    n_low = len(idx[order[-1]])                  

    l_c = R[idx[1]] 
    r_c = R[idx[2]] 
    t_stat, p_values = stats.ttest_ind(l_c, r_c, axis=0, equal_var=False)
    p_values = np.ma.filled(np.ma.asarray(p_values), 1.0)
    is_sig = np.flatnonzero(p_values < 0.05)     

    if ax is None:
        fig, (a0, a1) = plt.subplots(
            2, 1, figsize=(4.4, 4.6), sharex=True, constrained_layout=True,
            gridspec_kw=dict(height_ratios=[1.5, 1]))
    else:
        a0, a1 = ax
        fig = a0.figure

    # ---------------- raster ----------------
    ext = [centers[0], centers[-1], 0, len(row_order)]
    if raster == "heatmap":
        div = LinearSegmentedColormap.from_list(
            "div", ["#1d4ed8", "#f4f4f2", "#c2410c"])
        v = np.nanpercentile(np.abs(R), 99)
        v = v if np.isfinite(v) and v > 0 else 1.0
        im = a0.imshow(R[row_order], aspect="auto", origin="lower", cmap=div,
                       norm=TwoSlopeNorm(0, -v, v), extent=ext,
                       interpolation="nearest")
        if cbar:
            cb = fig.colorbar(im, ax=a0, fraction=.045, pad=.02)
            cb.set_label("z-scored ΔF/F", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    else:                                                  # dot raster
        thr = np.nanpercentile(R, 80)
        for r, t in enumerate(row_order):
            hit = np.flatnonzero(R[t] > thr)
            if len(hit):
                a0.plot(centers[hit], np.full(len(hit), r + .5), marker="|",
                        ls="none", ms=5, mew=1.2, color=COLOR[trial_cue[t]])
        a0.set_xlim(-3, 9); a0.set_ylim(0, len(row_order))

    a0.axhline(n_low, color=_INK, lw=1.2)
    for c, y in ((order[-1], n_low / 2),
                 (order[0], n_low + (len(row_order) - n_low) / 2)):
        a0.text(centers[0] + .03 * (centers[-1] - centers[0]), y, LABEL[c],
                color=COLOR[c], fontsize=9, weight="bold", va="center")
    a0.set_ylabel("trial")
    a0.set_title(title if title is not None else f"cell {cell}", fontsize=10)

    # ---------------- PSTH ----------------
    for c in order:
        sub = R[idx[c]]
        n_ok = np.sum(np.isfinite(sub), 0)
        m = np.nanmean(sub, 0)
        se = np.nanstd(sub, 0) / np.sqrt(np.maximum(n_ok, 1))
        m = np.where(n_ok >= 2, m, np.nan)                 
        a1.plot(centers, m, color=COLOR[c], lw=1.8, label=f"{LABEL[c]} (n={len(idx[c])})")
        a1.fill_between(centers, m - se, m + se, color=COLOR[c], alpha=.22, lw=0)
        a1.set_xlim(-3, 9); a1.set_ylim(-0.5, 1)

    if len(is_sig):
        a1.scatter(centers[is_sig], np.full(len(is_sig), 0.94),
        transform=a1.get_xaxis_transform(),
        marker='s', s=10, color=_INK, clip_on=False, zorder=5)
    a1.set_xlabel(xlabel); a1.set_ylabel("z-scored ΔF/F")
    a1.legend(frameon=False, fontsize=8)
    a1.spines[["top", "right"]].set_visible(False)

    for x in landmarks:
        a0.axvline(x, color=_INK, ls="--", lw=.9)
        a1.axvline(x, color=_INK, ls="--", lw=.9)

    if save:
        path = save if isinstance(save, str) else os.path.join(
            "spatial_raster_psth", f"cell_{cell}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=165, bbox_inches="tight")
    return fig, a0, a1
