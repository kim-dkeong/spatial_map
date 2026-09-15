import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

# maze range: -3~9
COLOR = {1: "#2f6fd0", 2: "#c0392b"}          # 1=left, 2=right
LABEL = {1: "Left", 2: "Right"}
_INK, _MUTED = "#1a1a1a", "#6b6b6b"

# 조건 정의: (key, 범례 라벨, 색, 선 스타일)
#   색 = cue 방향(파랑/빨강),  선 스타일 = 정답 여부(실선/점선)
#   -> 색 하나로 두 정보를 우겨넣지 않아 색각이상에서도 구분됩니다.
GROUP_SPECS = {
    "cue": [
        (1, "Left",  "#2f6fd0", "-"),
        (2, "Right", "#c0392b", "-"),
    ],
    "cue_outcome": [
        ("L_o", "Left · correct",  "#2f6fd0", "-"),
        ("R_o", "Right · correct", "#c0392b", "-"),
        ("L_x", "Left · error",    "#2f6fd0", "--"),
        ("R_x", "Right · error",   "#c0392b", "--"),
    ],
    "choice_outcome": [
        ("L_o", "Chose L · correct", "#2f6fd0", "-"),
        ("R_o", "Chose R · correct", "#c0392b", "-"),
        ("L_x", "Chose L · error",   "#2f6fd0", "--"),
        ("R_x", "Chose R · error",   "#c0392b", "--"),
    ],
}


def make_trial_groups(trials, ids, split="cue_outcome"):
    
    need = {"cue": ["cue"], "cue_outcome": ["cue", "correct"],
            "choice_outcome": ["choice", "correct"]}[split]
    for c in need + ["iTrial"]:
        if c not in trials.columns:
            raise KeyError(f"trials 에 '{c}' 컬럼이 없습니다. "
                           f"있는 컬럼: {list(trials.columns)}")

    t = trials.set_index("iTrial").reindex(np.asarray(ids))
    if t[need].isna().any().any():
        missing = np.asarray(ids)[t[need].isna().any(axis=1).to_numpy()]
        raise ValueError(f"trials 에 없는 trial 이 ids 에 있습니다: {missing[:10]}")

    if split == "cue":
        groups = t.cue.to_numpy().astype(int)
    else:
        side = (t.cue if split == "cue_outcome" else t.choice).to_numpy().astype(int)
        ok = t.correct.to_numpy().astype(int) > 0
        groups = np.array([f"{'L' if s == 1 else 'R'}_{'o' if c else 'x'}"
                           for s, c in zip(side, ok)], dtype=object)

    spec = [g for g in GROUP_SPECS[split] if np.any(groups == g[0])]
    dropped = [g[1] for g in GROUP_SPECS[split] if not np.any(groups == g[0])]
    if dropped:
        print(f"[groups] trial 이 0개라 제외: {', '.join(dropped)}")
    print("[groups] " + ",  ".join(
        f"{lab} n={int(np.sum(groups == k))}" for k, lab, _, _ in spec))
    return groups, spec


def _resolve_spec(groups, group_spec):
    if group_spec is not None:
        return [g for g in group_spec if np.any(groups == g[0])]
    uniq = set(np.unique(groups).tolist())
    for name, spec in GROUP_SPECS.items():
        if uniq <= {g[0] for g in spec}:
            return [g for g in spec if np.any(groups == g[0])]
    # 모르는 라벨 -> 기본 팔레트로 자동 배정
    pal = ["#2f6fd0", "#c0392b", "#1c7a5b", "#a8600d", "#6b46c1"]
    return [(k, str(k), pal[i % len(pal)], "-")
            for i, k in enumerate(sorted(uniq, key=str))]


def plot_raster_psth(maps, centers, trial_group, cell, ax=None, raster="heatmap",
                     landmarks=(0.0, 4.0), title=None, cbar=True,
                     group_spec=None, save=None, xlabel="z − cue z (VR unit)",
                     stat=None, alpha=0.05, order=None, legend_loc="best"):
    
    maps = np.asarray(maps)
    centers = np.asarray(centers)
    trial_group = np.asarray(trial_group)
    if maps.shape[1] != len(trial_group):
        raise ValueError(f"trial 수가 안 맞습니다: maps {maps.shape[1]}, "
                         f"trial_group {len(trial_group)}. spatial_map 이 반환한 "
                         f"ids 순서로 만들었는지 확인하세요.")
    R = maps[cell]                                        # (n_trials, n_bins)

    if order is not None and group_spec is None:          # 구버전 호환
        group_spec = [g for g in GROUP_SPECS["cue"] if g[0] in order]
    spec = _resolve_spec(trial_group, group_spec)
    idx = {k: np.flatnonzero(trial_group == k) for k, *_ in spec}

    # raster 행 순서: spec 의 첫 조건이 맨 위
    row_order = np.concatenate([idx[k] for k, *_ in reversed(spec)])
    bounds, acc = [], 0                                   # 조건 경계선 y 좌표
    for k, *_ in reversed(spec):
        acc += len(idx[k])
        bounds.append(acc)

    if ax is None:
        fig, (a0, a1) = plt.subplots(
            2, 1, figsize=(4.8, 5.2), sharex=True, constrained_layout=True,
            gridspec_kw=dict(height_ratios=[1.6, 1]))
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
        col_of = {k: c for k, _, c, _ in spec}
        for r, t in enumerate(row_order):
            hit = np.flatnonzero(R[t] > thr)
            if len(hit):
                a0.plot(centers[hit], np.full(len(hit), r + .5), marker="|",
                        ls="none", ms=4, mew=1.0, color=col_of[trial_group[t]])
        a0.set_xlim(centers[0], centers[-1]); a0.set_ylim(0, len(row_order))

    # 조건 경계선 + 좌측 라벨
    lo = 0
    for (k, lab, c, _), hi in zip(reversed(spec), bounds):
        if hi < len(row_order):
            a0.axhline(hi, color=_INK, lw=1.1)
        a0.text(centers[0] + .02 * (centers[-1] - centers[0]), (lo + hi) / 2,
                f"{lab}  n={len(idx[k])}", color=c, fontsize=7.5,
                weight="bold", va="center")
        lo = hi
    a0.set_ylabel("trial")
    a0.set_title(title if title is not None else f"cell {cell}", fontsize=10)

    # ---------------- PSTH ----------------
    for k, lab, c, ls in spec:
        sub = R[idx[k]]
        if len(sub) == 0:
            continue
        n_ok = np.sum(np.isfinite(sub), 0)
        m = np.nanmean(sub, 0)
        se = np.nanstd(sub, 0) / np.sqrt(np.maximum(n_ok, 1))
        m = np.where(n_ok >= 2, m, np.nan)
        a1.plot(centers, m, color=c, ls=ls, lw=1.8,
                label=f"{lab} (n={len(idx[k])})")
        if len(sub) >= 2:
            a1.fill_between(centers, m - se, m + se, color=c, alpha=.16, lw=0)

    # ---------------- 두 조건 t-test ----------------
    if stat:
        ka, kb = (spec[0][0], spec[1][0]) if stat is True else stat
        a_, b_ = R[idx[ka]], R[idx[kb]]
        if len(a_) >= 2 and len(b_) >= 2:
            _, p = stats.ttest_ind(a_, b_, axis=0, equal_var=False,
                                   nan_policy="omit")
            p = np.ma.filled(np.ma.asarray(p), 1.0)
            sig = np.flatnonzero(p < alpha)
            if len(sig):
                a1.scatter(centers[sig], np.full(len(sig), 1.03),
                           transform=a1.get_xaxis_transform(), marker="s",
                           s=13, color=_INK, clip_on=False, zorder=5)
            a1.text(0.01, 0.98, f"sig bar: {ka} vs {kb}, p<{alpha} — "
                    f"{len(sig)}/{len(p)} bins", transform=a1.transAxes,
                    fontsize=7, va="top", color=_MUTED)

    a1.set_xlabel(xlabel); a1.set_ylabel("z-scored ΔF/F")
    a1.legend(frameon=False, fontsize=7.5, loc=legend_loc,
              ncol=2 if len(spec) > 2 else 1)
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
