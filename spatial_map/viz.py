"""spatial_map() 결과를 raster + PSTH 그림으로 그리는 함수."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# maze range: -3~9
COLOR = {1: "#2f6fd0", 2: "#c0392b"}          # 1=left, 2=right
LABEL = {1: "Left", 2: "Right"}
_INK, _MUTED = "#1a1a1a", "#6b6b6b"


def plot_raster_psth(maps, centers, trial_cue, cell, ax=None, raster="heatmap",
                     landmarks=(0.0, 4.0), title=None, cbar=True,
                     order=(2, 1), save=None, xlabel="z − cue z (VR unit)"):
    """
    spatial_map() 결과를 left/right Raster-PSTH 로 그린다.

        maps, occ, edges, centers, ids = spatial_map(Z, F, valid)
        cue_of = T.set_index("iTrial").cue.reindex(ids).to_numpy()
        plot_raster_psth(maps, centers, cue_of, cell=4)

    파라미터
      maps      : (n_cells, n_trials, n_bins)   ← spatial_map 의 첫 반환값
      centers   : (n_bins,)                     ← spatial_map 의 네 번째 반환값
      trial_cue : (n_trials,)  각 trial 의 조건. 1=left, 2=right.
                  spatial_map 이 반환한 ids 순서와 **같은 순서**여야 한다.
      cell      : 그릴 세포 인덱스 (maps 의 0번 축)
      ax        : (ax_raster, ax_psth) 튜플. None 이면 새 figure 를 만든다.
      raster    : "heatmap" (칸별 평균 색) | "dot" (임계 넘은 bin 만 점)
      landmarks : 세로 점선을 그을 x 좌표들. 기본 (0.0,) = cue
      order     : raster 에서 위→아래 조건 순서. 기본 (2,1) = Right 위, Left 아래
      save      : True/파일경로. 지정하면 ./spatial_raster_psth/cell_{cell}.png 로 저장
                  (경로 문자열을 넘기면 그 경로에 저장, 디렉터리는 자동 생성)

    반환: (fig, ax_raster, ax_psth)
    """
    maps = np.asarray(maps)
    trial_cue = np.asarray(trial_cue)
    if maps.shape[1] != len(trial_cue):
        raise ValueError(f"trial 수가 안 맞습니다: maps {maps.shape[1]}, "
                         f"trial_cue {len(trial_cue)}. spatial_map 이 반환한 "
                         f"ids 순서로 trial_cue 를 만들었는지 확인하세요.")
    R = maps[cell]                                        # (n_trials, n_bins)

    idx = {c: np.flatnonzero(trial_cue == c) for c in order}
    row_order = np.concatenate([idx[c] for c in reversed(order)])  # 아래가 먼저
    n_low = len(idx[order[-1]])                           # 구분선 위치

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
        m = np.where(n_ok >= 2, m, np.nan)                 # trial 2개 미만은 생략
        a1.plot(centers, m, color=COLOR[c], lw=1.8, label=f"{LABEL[c]} (n={len(idx[c])})")
        a1.fill_between(centers, m - se, m + se, color=COLOR[c], alpha=.22, lw=0)
        a1.set_xlim(-3, 9); a1.set_ylim(-0.5, 1)
    a1.set_xlabel(xlabel); a1.set_ylabel("z-scored ΔF/F")
    a1.legend(frameon=False, fontsize=8)
    a1.spines[["top", "right"]].set_visible(False)

    for x in landmarks:
        a0.axvline(x, color=_INK, ls="--", lw=.9)
        a1.axvline(x, color=_INK, ls="--", lw=.9)

    if save:
        # save=True 면 기본 경로, 문자열이면 그 경로에 저장
        path = save if isinstance(save, str) else os.path.join(
            "spatial_raster_psth", f"cell_{cell}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=165, bbox_inches="tight")
    return fig, a0, a1
